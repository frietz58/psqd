from enum import Enum


class PointTasks(Enum):
    obstacle = "Obstacle"
    top_reach = "TopReach"
    side_reach = "SideReach"
    obstacle_top = "ObstacleTop"


class PointMazeTasks(Enum):
    obstacle = "Obstacle"
    fire_room = "FireRoom"
    coin = "Coin"


class PandaTasks(Enum):
    reach = "Reach"
    avoid = "Avoid"


