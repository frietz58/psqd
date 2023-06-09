import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import random
import os

from misc.buffer import ReplayBuffer
from misc.utils import asvgd_rbf_kernel, adaptive_isotropic_gaussian_kernel, min_log_th

from enum import Enum


class PSQDModes(Enum):
    IS = "ImportanceSampling"
    ASVGD = "AmortizedStein"


class PSQD:
    """
    Our main algorithm, Prioritized Soft Q-Decomposition (PSQD).
    It takes a a list of Q (and optionally policy) networks, we always assume the last indext of the list refers to the
    current task, while the previous indices refer to networks trained for previous tasks. We only ever change the networks
    for the current task, the previous networks are kept fixed.

    The priority thresholds are the \epsilon_i scalars in the paper, they determine the allowed value divergence
    from the optimal action value for each previous task.

    Our implementation supports two modes: Importance Sampling (IS) and Amortized Stein Variational Gradient Descent (ASVGD).
    IS-mode can be used to directly sample from the Q-function when the action space is of low dimensionality.
    ASCGD-mode corresponds to the SQL method proposed by Haarnoja et al and corrsponds to learning a sampling network
    for the Q-function, which scales better to high-dimensional action spaces.

    """
    def __init__(self,
                 q_nets: list[nn.Module],
                 q_net_targets: list[nn.Module],
                 replay_buffer: ReplayBuffer,
                 device: torch.device,
                 mode: PSQDModes = PSQDModes.IS,
                 asvgd_nets: list[nn.Module] = [],
                 priority_thresholds: list[float] = [],
                 action_size: int = 2,
                 n_particles: int = 32,
                 q_lr: float = 0.001,
                 pi_lr: float = 0.0001,
                 weight_decay: float = 0.0,
                 reward_scale: int = 1,
                 gamma: float = 1.0,
                 loss_fn_str: str = "MSELoss",
                 tau: float = 0.0,
                 hard_freq: int = 10000,
                 lr_exponential_decay: float = 0.99,
                 tensorboard_writer: SummaryWriter = None,
                 zeta_dim: int = -1,
                 grad_clip: bool = False,
                 seed=None
                 ):
        self.action_size = action_size
        self.pi_nets = asvgd_nets
        self.pi_optim = None
        if len(self.pi_nets) > 0:
            self.pi_optim = Adam(self.pi_nets[-1].parameters(), lr=pi_lr, weight_decay=weight_decay) if len(self.pi_nets) > 0 else None
        if self.pi_optim is not None:
            self.pi_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.pi_optim, gamma=lr_exponential_decay)
        self.q_nets = q_nets
        self.q_optim = Adam(self.q_nets[-1].parameters(), lr=q_lr, weight_decay=weight_decay)
        self.q_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.q_optim, gamma=lr_exponential_decay)
        self.q_net_targets = q_net_targets
        self.priority_thresholds = priority_thresholds
        self.buffer = replay_buffer
        self.device = device
        self.n_particles = n_particles
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.tau = tau
        self.hard_freq = hard_freq
        self.tb_writer = tensorboard_writer
        self.mode = mode
        if zeta_dim < 0:
            self.zeta_dim = action_size
        else:
            self.zeta_dim = zeta_dim
        self.grad_clip = grad_clip

        if self.mode == PSQDModes.ASVGD:
            assert len(self.pi_nets) > 0, "ASVGD mode requires a policy network"

        if loss_fn_str == "MSELoss":
            self.loss_fn = nn.MSELoss(reduction="mean")
        elif loss_fn_str == "L1Loss":
            self.loss_fn = nn.SmoothL1Loss(reduction="mean")
        else:
            raise ValueError(f"Loss '{loss_fn_str}' not recognized")

        self.q_update_counter = 0
        self.pi_update_counter = 0

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _make_q_inp(self, state_tensor, batchsize, action_range=1, grid_res=20, do_repeat_states=True):
        """
        Helper function for Importance Sampling action selection.
        Creates a grid of actions covering the action space, and repeats the grid for each state in the (batch) state tensor.
        :param state_tensor: the (batch of) states
        :param batchsize: number of states in the batch / state_tensor
        :param action_range: the range of actions to cover (1 range -> -1 - +1)
        :param grid_res: resolution of the grid in each dimension (range / grid_res = step size)
        :param do_repeat_states: whether to repeat the state tensor for each action in the grid
        :return: the input to the q network, which is a tensor of concatenated states and actions, and the grid of actions
        """
        x_, y_ = np.meshgrid(np.linspace(-action_range, action_range, grid_res),
                             np.linspace(action_range, -action_range, grid_res))
        actions = np.stack((x_, y_), axis=-1)
        actions = actions.reshape(-1, self.action_size)
        actions = torch.from_numpy(actions).to(self.device).to(torch.float32)
        actions += torch.normal(mean=torch.zeros(grid_res ** 2, 2), std=torch.ones(grid_res ** 2, 2) * 0.1).to(self.device)  # add some noise to the grid
        if do_repeat_states:
            states_ = state_tensor.unsqueeze(1).repeat(1, grid_res * grid_res, 1)
        else:
            states_ = state_tensor
        q_inp = torch.cat((states_, actions.unsqueeze(0).repeat(batchsize, 1, 1)), 2).to(torch.float)
        return q_inp, actions

    def is_act(
            self,
            state_tensor: torch.Tensor,
            actions_per_state: int = 1,
            do_repeat_states: bool = True,
            **kwargs
    ):
        """
        Importance Sampling action selection via a grid over the action space and reweighting points in the grid
        according to the Q function.

        In unconstrained setting this is just a shortcut for SQL that avoids learning the policy network (obviously
        does not scale to high dimensions).
        In constrained setting we perform IS w.r.t the prioritized q values, aka constraint-violating actions have zero
        probability of being sampled, while all policy networks serve as proposal distributions.

        :param state_tensor: The tensor of states to select actions for
        :param actions_per_state: How many actions to select per state
        :param do_repeat_states: Whether to repeat the state tensor needs to be repeated for each action
        :return: Actions for each state in state_tensor
        """
        batchsize = state_tensor.shape[0]
        if len(self.q_nets) == 1:
            q_inp, action_grid = self._make_q_inp(state_tensor, batchsize, do_repeat_states=do_repeat_states)
            q = self.q_nets[-1](q_inp.to(self.device))
            # q += torch.ones_like(q)
            is_weights = torch.exp(q.double()).flatten()
        else:
            q_inp, action_grid = self._make_q_inp(state_tensor, batchsize, do_repeat_states=do_repeat_states)
            state_tensor = q_inp[:, :, :-self.action_size]
            action_tensor = q_inp[:, :, -self.action_size:]
            pref_q, constraint_prod, _, _ = self.check_constraints(state_tensor, action_tensor, use_target=False)

            is_weights = torch.exp(pref_q.to(torch.double)).flatten()
            is_weights *= constraint_prod.flatten()  # forbidden actions get zero weight to be sampled by IS
            is_weights[torch.isnan(is_weights)] = 0
            q = pref_q + min_log_th(constraint_prod)

        # for each state, draw n_particle * argmax_k actions and used the best of the argmax_k ones
        argmax_k = 1
        k_action_particle_idxs = torch.multinomial(
            is_weights,
            num_samples=batchsize * actions_per_state * argmax_k,
            replacement=True
        )
        particle_q_vals = q.flatten()[k_action_particle_idxs]
        particle_q_vals = particle_q_vals.reshape((batchsize, actions_per_state, argmax_k))
        max_idxs = particle_q_vals.argmax(dim=2).unsqueeze(-1)  # select index of index of (no typo) best action per state

        particle_idxs = k_action_particle_idxs.reshape((batchsize, actions_per_state, argmax_k))
        action_idxs = torch.gather(particle_idxs, dim=2, index=max_idxs)  # select index of best action per state

        actions_ = action_grid.unsqueeze(0).repeat(batchsize, 1, 1).reshape(-1, self.action_size)[action_idxs.flatten()]
        actions_ = actions_.reshape(batchsize, actions_per_state, self.action_size)  # select best action per state

        return actions_

    def asvgd_act(
            self,
            state_tensor: torch.Tensor,
            actions_per_state: int = 1,
            do_repeat_states: bool = True,
            **kwargs
    ):
        """
        Action selection using the ASVGD policy sampling network.

        In the vanilla SQL/unconstrainted case, we simply sample from the policy network.
        In the prioritized/constrained case, we sample use the policy networks as proposal distributions and perform
        importance sampling w.r.t the prioritized Q-value, such that actions outside of the action indifference space
        have zero probability of being sampled.

        :param state_tensor: The tensor of states to for which we want to select actions
        :param actions_per_state: How many actions to sample per state
        :param do_repeat_states: In the constrained case, whether the state tensor has already been or needs to be
            repeated for once for each policy (our sampling procedure samples n_actions from each policy and then
            performance importance sampling w.r.t the prioritized Q-value)
        :return: Actions for each state in the state_tensor
        """
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)  # we always want a batch dimension...

        if len(self.pi_nets) == 1:
            # no other policies / unconstrained, simply act according to policy
            zeta = torch.normal(mean=0.0, std=1.0, size=(state_tensor.shape[0], state_tensor.shape[1], self.action_size)).to(self.device)
            inp = torch.cat((state_tensor, zeta), dim=-1)
            inp = inp.view(-1, state_tensor.shape[-1] + self.action_size)

            action = self.pi_nets[-1].forward(inp.to(torch.float32))

        elif len(self.pi_nets) > 1:
            # constrained, act according to task-priority constrained action space
            if do_repeat_states:
                state_tensor = state_tensor.repeat(self.n_particles, 1, 1).movedim(1, 0)

            zeta = torch.normal(mean=0.0, std=1.0, size=(state_tensor.shape[0], state_tensor.shape[1], self.action_size)).to(
                self.device)
            inp = torch.cat((state_tensor, zeta), dim=-1)
            inp = inp.view(-1, state_tensor.shape[-1] + self.action_size)

            all_actions = []
            for policy in self.pi_nets:
                all_actions.append(policy.forward(inp.to(torch.float32)))

            all_actions = torch.stack(all_actions, dim=1).view(-1, len(self.pi_nets) * self.n_particles, self.action_size)

            state_tensor = state_tensor.repeat_interleave(len(self.pi_nets), dim=1)
            pref_q, constraint_prod, _, _ = self.check_constraints(state_tensor, all_actions, use_target=False)

            # IS action based on preference q value and constraint product
            is_weights = torch.exp(pref_q.to(torch.double)).flatten()
            is_weights *= constraint_prod.squeeze().flatten()
            is_weights[torch.isnan(is_weights)] = 0
            k_action_particle_idxs = torch.multinomial(is_weights.squeeze(), num_samples=actions_per_state * state_tensor.shape[0], replacement=True)

            action = torch.index_select(all_actions.reshape(-1, self.action_size), index=k_action_particle_idxs, dim=0)
            action = action.reshape(state_tensor.shape[0], actions_per_state, self.action_size)

        return action

    def check_constraints(
            self,
            state_tensor: torch.Tensor,
            action_tensor: torch.Tensor,
            use_target: bool = False
    ):
        """
        Given states and per state actions, checks which of the actions are allowed by the n-1 higher priority tasks.
        For each state, we calculate the priority level, aka which tasks expresses its preference over actions.
        This *should* always be the lowest priority task, but it could theoretically happend that we have not sampled an action in the global indifference sapce, then we use the next lowest task to assign the q-value.
        All taks higher tasks than the priority level form the action-indifference space and mask out forbidden actions.

        Below, we calculate the Q-values of every task and the constraint-indicator functions for the n-1 higher priority tasks.
        We then take the product of constraint indicator functions to find the priority level for each state.
        We return the q values of actions at the obtained priority level and the corresponding product of constraint
        indicator functions, aka the action indifference space for each state.

        The Q-values for the prioritied task, as define in the paper, can than be obtained by adding the log of the
        constraint product to the preference q values.

        :param state_tensor: the (batchsize, per_state_actions, state_dim)-shaped tensor of states
        :param action_tensor: the (batchsize, per_state_actions, action_dim)-shaped, tensor of actions
        :param use_target: whether to use the target networks for the q values
        :return:
        """
        batchsize = state_tensor.shape[0]
        n_components = len(self.q_nets)
        per_state_actions = action_tensor.shape[1]
        q_inp = torch.cat((state_tensor, action_tensor), dim=-1)
        q_inp = q_inp.reshape(-1, state_tensor.shape[-1] + action_tensor.shape[-1])  # reshape for NN forward pass

        qs = torch.zeros(batchsize, per_state_actions, n_components).to(self.device)  # q levels for all tasks
        cs = torch.ones(batchsize, per_state_actions, n_components).to(self.device)  # constrain indications for n-1 highest priority tasks
        cs_soft_grad = torch.zeros(batchsize, per_state_actions, n_components).to(self.device)  # soft, differentiable version of constraint indicator, needed for ASVGD sampler
        c_prods = torch.ones(batchsize, per_state_actions, 1).to(self.device)  # product of n-1 constrain indicators
        c_prod_soft = torch.ones(batchsize, per_state_actions, 1).to(self.device)  # product of n-1 constrain indicators
        c_prod_lvls = torch.ones(batchsize, per_state_actions, n_components).to(self.device)  # product of n-1 constrain indicators
        c_prod_lvls_soft = torch.ones(batchsize, per_state_actions, n_components).to(self.device)  # product of n-1 constrain indicators
        per_state_prio_level = torch.zeros(batchsize, 1, dtype=torch.int64).to(self.device)  # lowest allowed task priorty in each state

        # starting with highest, get q for each task priority level
        for i in range(len(self.q_nets)):
            if use_target:
                q = self.q_net_targets[i](q_inp.to(torch.float32))
            else:
                q = self.q_nets[i](q_inp.to(torch.float32))

            # reshape q from (batchsize x n_components, state_dim) to (batchsize, per_state_actions, 1)
            q = q.view(batchsize, per_state_actions, 1)
            qs[:, :, i] = q.squeeze(-1)

            if i == n_components - 1:
                # for lowest prior q we do need to calc constraint function, we just need the q values
                break

            # find the best action q value IN EACH STATE (hence the above view)
            best_q = q.max(dim=1, keepdim=True)[0]
            # and calc per state divergence from best q for each action
            divergence = q - best_q.repeat(1, per_state_actions, 1)

            # calc constrain indicator function given divergences and allowed thresholds
            with torch.no_grad():  # since heavisude has no deriv, we must exlude it from forward pass...
                c = torch.heaviside(divergence + self.priority_thresholds[i], torch.tensor(1, dtype=torch.float32))
                cs[:, :, i] = c.squeeze(-1)

            c_soft = (torch.ones(1, device=self.device) / torch.pi) * torch.atan2(torch.ones_like(divergence) * 0.01, - (divergence + self.priority_thresholds[i]))
            assert c_soft.shape == c.shape
            cs_soft_grad[:, :, i] = c_soft.squeeze(-1)

            c_prods = c_prods * c  # prod of constraint indicator funcs up to i... 1 if all constraints so far are met
            c_prod_soft = c_prod_soft * c_soft

            c_prod_lvls[:, :, i] = c_prods.squeeze(-1)
            c_prod_lvls_soft[:, :, i] = c_prod_soft.squeeze(-1)
            # increment prio level per state IFF all constraints are met
            per_state_prio_level += c_prods.max(dim=1).values.to(torch.int64)

        # select q values and constraint indicator product depending on priority level
        prio_lvl_idx = per_state_prio_level.unsqueeze(-1).repeat(1, per_state_actions, 1)
        pref_q = qs.gather(dim=2, index=prio_lvl_idx).squeeze(-1)
        last_q = qs[:, :, -1]
        assert last_q.shape == pref_q.shape
        prio_lvl_idx_c = prio_lvl_idx - 1  # constrain level is priority level - 1
        prio_lvl_idx_c[prio_lvl_idx_c < 0] = 0  # aviod underflow/wraparound
        constraint_prod = c_prod_lvls.gather(dim=2, index=prio_lvl_idx_c).squeeze(-1)
        constraint_prod_soft = c_prod_lvls_soft.gather(dim=2, index=prio_lvl_idx_c).squeeze(-1)

        if len(self.q_nets) == 1:
            # unconstrained, just return q values and all constraints are met
            pref_q = qs.squeeze(-1)
            constraint_prod = torch.ones_like(pref_q)

        return pref_q, constraint_prod, constraint_prod_soft, last_q

    def act(self,
            state_tensor:  torch.Tensor,
            actions_per_state: int = 1,
            do_repeat_states: bool = True,
            **kwargs
            ):
        """
        Act method interface for the SQL agent.
        Either we act according to IS over the action space (shortcut) or use sampling network (vanilla SQL alg).
        :param state_tensor: The tensor of states for which we want to get actions
        :param actions_per_state: How many action to get for each state
        :param do_repeat_states: Helper flag whether the state tensor needs to be repeated for each policy or action...
        :param kwargs:
        :return:
        """
        if self.mode == PSQDModes.IS:
            return self.is_act(state_tensor, actions_per_state, do_repeat_states)
        elif self.mode == PSQDModes.ASVGD:
            return self.asvgd_act(state_tensor, actions_per_state, do_repeat_states)
        else:
            raise NotImplementedError

    def update(self):
        """
        Update method interface for the SQL agent.
        Depending on mode, we update only or soft Q function via TD or additionally policy via ASVGD.
        :return:
        """
        if len(self.buffer) < self.buffer.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample()

        self.td_update(states, actions, rewards, next_states, dones)
        if self.mode == PSQDModes.ASVGD:
            self.asvgd_update(states, actions, rewards, next_states, dones)

    def step_lr(self):
        self.pi_scheduler.step()
        self.q_scheduler.step()
        self.tb_writer.add_scalar('lr/pi', self.pi_scheduler.get_lr()[0], self.pi_update_counter)
        self.tb_writer.add_scalar('lr/q', self.q_scheduler.get_lr()[0], self.q_update_counter)

    def td_update(self, states, actions, rewards, next_states, dones):
        """
        TD update for the off-policy, optimal soft Q-function.
        When we only have one Q-net its vanilla SQL algorithm (aka pre-training in our paper).
        When we have more than one Q-net, we are in the constrained/task-priority setting. In this case we update
        only the last Q-net (we assume all previous Q-nets are already trained) towards the constrained/prioritized
        constituent Q-function, aka the softmax target value is over the action indifference space, instead of over the
        entire action space as in the unconstrained setting.
        """
        self.q_optim.zero_grad()

        # predict q for batch
        batch_q = self.q_nets[-1].forward(torch.cat((states, actions), dim=1))
        self.tb_writer.add_scalar("train/q_mean", batch_q.mean().item(), self.q_update_counter)
        self.tb_writer.add_scalar("train/q_var", batch_q.var().item(), self.q_update_counter)

        with torch.no_grad():
            if len(self.q_nets) == 1:
                # vanilla SQL. sample n_particle actions per state and get q_value
                next_states_ = next_states.repeat(self.n_particles, 1, 1).movedim(1, 0)
                proposal_actions = self.act(state_tensor=next_states_, actions_per_state=self.n_particles, do_repeat_states=False)
                # proposal_actions = proposal_actions.reshape(next_states_.shape)
                proposal_actions = proposal_actions.reshape(next_states_.shape[0], next_states_.shape[1], -1)
                q_inp = torch.cat((next_states_, proposal_actions), dim=2)
                q_soft = self.q_net_targets[-1].forward(q_inp)
                q_soft = q_soft.squeeze(-1)

            elif len(self.q_nets) > 1:
                # constrained/prioritized SQL. Action samples outside of the action indifference space get value -inf
                # in q_soft. Notice the softmax only essentially remove the -inf values again and only takes keeps the best...
                next_states_ = next_states.repeat(self.n_particles, 1, 1).movedim(1, 0)
                proposal_actions = self.act(state_tensor=next_states_, actions_per_state=self.n_particles, do_repeat_states=False)
                # pref_q, constraint_prod, _, _ = self.check_constraints(next_states_, proposal_actions, use_target=True)
                _, constraint_prod, _, pref_q = self.check_constraints(next_states_, proposal_actions, use_target=True)
                q_soft = pref_q + min_log_th(constraint_prod)

            else:
                raise ValueError("self.q_nets must have length >= 1")

            # calulate soft value from for next_states from q_soft (eq 10 in Haarnoja SQL paper)
            assert q_soft.shape == (self.buffer.batch_size, self.n_particles)
            v_soft = torch.logsumexp(q_soft, dim=1, keepdim=True)

            # importance weighting just adds a constant term in linear scale
            v_soft -= torch.log(torch.tensor([self.n_particles])).to(self.device)

            # Next line can be found in SQL reference implementation, but why?
            # v_soft += torch.log(torch.tensor([2]).to(self.device)) * self.action_size
            # this seems to be a misstake, with this line soft q values are greatly overstimating the true q values...

            target = self.reward_scale * rewards + self.gamma * v_soft * (1 - dones)

        loss = self.loss_fn(batch_q, target)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q_nets[-1].parameters(), 1)

        self.q_optim.step()

        self.tb_writer.add_scalar("train/q_loss", loss.item(), self.q_update_counter)
        self.q_update_counter += 1

        # polyak soft target network udpate
        for target_param, current_param in zip(self.q_net_targets[-1].parameters(), self.q_nets[-1].parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1.0 - self.tau) * target_param.data)

        # hard target network update
        if self.q_update_counter % self.hard_freq == 0:
            print("target_update")
            self.q_net_targets[-1].load_state_dict(self.q_nets[-1].state_dict())

    def asvgd_update(self, states, actions, rewards, next_state, dones):
        """
        Just the Amortized Stein Variational Gradient Descent update for the policy sampling network in SQL.
        In the constrained/prioritized setting, we update the sampling network towards the prioritized Q-function,
        aka we learn to sample actions from the action indifference space, since the prioritized policy has zero
        probability. This requires that the constraint indicator function is differentiable.

        This code is, essentially, taken from Haarnoja's SQL reference implementation.
        """
        batch_size = states.shape[0]
        state_tensor = states.unsqueeze(1).repeat(1, self.n_particles * 2, 1)
        state_tensor_2 = states.unsqueeze(1).repeat(1, self.n_particles, 1)

        # 1) sample actions...
        zeta_1 = torch.normal(mean=0, std=1, size=(batch_size, self.n_particles * 2, self.action_size)).to(
            self.device)
        actions = self.pi_nets[-1].forward(torch.cat((state_tensor, zeta_1), dim=2))
        assert actions.shape == (batch_size, self.n_particles * 2, self.action_size)

        # SVGD requires computing two empirical expectations over actions
        #  To that end, we first sample a single set of actions, and later split them into two sets.
        n_updated_actions = int(self.n_particles)
        n_fixed_actions = int(self.n_particles)

        fixed_actions, updated_actions = torch.split(
            actions, [n_fixed_actions, n_updated_actions], dim=1)
        fixed_actions = fixed_actions.detach()
        fixed_actions.requires_grad = True  # needed for autograd
        assert fixed_actions.shape == torch.Size([batch_size, n_fixed_actions, self.action_size])
        assert updated_actions.shape == torch.Size([batch_size, n_updated_actions, self.action_size])

        _, _, constraint_prod_soft, pref_q = self.check_constraints(state_tensor_2, fixed_actions, use_target=False)
        svgd_target_values = pref_q + min_log_th(constraint_prod_soft)

        # Change of variables squash correction for tanh in policy network forward pass.
        squash_correction = torch.sum(
            torch.log(1 - fixed_actions ** 2 + 1e-6), dim=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = torch.autograd.grad(log_p.sum(), fixed_actions, create_graph=True)[0]
        grad_log_p = grad_log_p.unsqueeze(2).data
        grad_log_p.requires_grad = False
        grad_log_p = grad_log_p.detach()
        assert grad_log_p.shape == torch.Size([batch_size, n_fixed_actions, 1, self.action_size])

        # kernel function and gradient for the SVGD update
        kappa, kappa_grad = adaptive_isotropic_gaussian_kernel(xs=fixed_actions, ys=updated_actions)
        kappa = kappa.unsqueeze(3)
        assert kappa.shape == torch.Size([batch_size, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient:
        action_gradients = torch.mean(
            kappa * grad_log_p + kappa_grad, dim=1)
        assert action_gradients.shape == torch.Size([batch_size, n_updated_actions, self.action_size])

        # Propagate the gradient through the policy network.
        gradients = torch.autograd.grad(
            outputs=updated_actions,
            inputs=self.pi_nets[-1].parameters(),
            grad_outputs=action_gradients,
            create_graph=True)

        surrogate_loss = sum([
            torch.sum(w * Variable(g.data, requires_grad=False))
            for w, g in zip(self.pi_nets[-1].parameters(), gradients)
        ])
        self.tb_writer.add_scalar("train/asvgd_loss", surrogate_loss.item(), self.pi_update_counter)

        self.pi_optim.zero_grad()
        (-surrogate_loss).backward()

        self.pi_optim.step()
        self.pi_update_counter += 1
