from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class KeyDoorTreasureEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=15, max_step=100):
        super().__init__(
            grid_size=size,
            max_steps=max_step
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, 1)

        # Create a vertical splitting wall
        self.grid.vert_wall((width-1)//2, 0, length = 11)
        self.grid.vert_wall((width-1)//2, 12, length = 2)
        # Create a horizontal wall
        self.grid.horz_wall(0,(height-1)//2, length = 3)
        self.grid.horz_wall(4,(height-1)//2, length = 7)
        self.grid.horz_wall(12,(height-1)//2, length = 2)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.put_agent(1, height-2, 0)

        # Place a door in the wall
        self.put_obj(Door('yellow', is_locked=True), (width-1)//2, 11)

        # Place a yellow key on the left side
        self.put_obj(Key('yellow'), 6, 1)

        self.mission = "use the key to open the door and then get to the goal"

class KeyDoorTreasureEnv5x5(KeyDoorTreasureEnv):
    def __init__(self):
        super().__init__(size=5, max_step=100)

class KeyDoorTreasureEnv6x6(KeyDoorTreasureEnv):
    def __init__(self):
        super().__init__(size=6, max_step=100)

class KeyDoorTreasureEnv16x16(KeyDoorTreasureEnv):
    def __init__(self):
        super().__init__(size=16, max_step=100)

register(
    id='MiniGrid-KeyDoorTreasure-5x5-v0',
    entry_point='gym_minigrid.envs:KeyDoorTreasureEnv5x5'
)

register(
    id='MiniGrid-KeyDoorTreasure-6x6-v0',
    entry_point='gym_minigrid.envs:KeyDoorTreasureEnv6x6'
)

register(
    id='MiniGrid-KeyDoorTreasure-8x8-v0',
    entry_point='gym_minigrid.envs:KeyDoorTreasureEnv'
)

register(
    id='MiniGrid-KeyDoorTreasure-16x16-v0',
    entry_point='gym_minigrid.envs:KeyDoorTreasureEnv16x16'
)
