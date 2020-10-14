import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX


class ColRowTrajectory(gym.core.Wrapper):
    """
    we change the actions to be 0:N, 1:E, 2:S, 3:W
    """

    def __init__(self, env):
        super().__init__(env)
        self.Trajectories = [((1, 1) , 2 , 0 , (1, 2) , False , {}) , ((1, 2) , 2 , 0 , (1, 3) , False , {}) ,
                        ((1, 3) , 2 , 0 , (1, 4) , False , {}) , ((1, 4) , 2 , 0 , (1, 5) , False , {}) ,
                        ((1, 5) , 2 , 0 , (1, 6) , False , {}) , ((1, 6) , 2 , 0 , (1, 6) , False , {}) ,
                        ((2, 1) , 2 , 0 , (2, 2) , False , {}) , ((2, 2) , 2 , 0 , (2, 3) , False , {}) ,
                        ((2, 3) , 2 , 0 , (2, 4) , False , {}) , ((2, 4) , 2 , 0 , (2, 5) , False , {}) ,
                        ((2, 5) , 2 , 0 , (2, 6) , False , {}) , ((2, 6) , 2 , 0 , (2, 6) , False , {}) ,
                        ((3, 1) , 2 , 0 , (3, 2) , False , {}) , ((3, 2) , 2 , 0 , (3, 3) , False , {}) ,
                        ((3, 3) , 2 , 0 , (3, 4) , False , {}) , ((3, 4) , 2 , 0 , (3, 5) , False , {}) ,
                        ((3, 5) , 2 , 0 , (3, 6) , False , {}) , ((3, 6) , 2 , 0 , (3, 6) , False , {}) ,
                        ((4, 1) , 2 , 0 , (4, 2) , False , {}) , ((4, 2) , 2 , 0 , (4, 3) , False , {}) ,
                        ((4, 3) , 2 , 0 , (4, 4) , False , {}) , ((4, 4) , 2 , 0 , (4, 5) , False , {}) ,
                        ((4, 5) , 2 , 0 , (4, 6) , False , {}) , ((4, 6) , 2 , 0 , (4, 6) , False , {}) ,
                        ((5, 1) , 2 , 0 , (5, 2) , False , {}) , ((5, 2) , 2 , 0 , (5, 3) , False , {}) ,
                        ((5, 3) , 2 , 0 , (5, 4) , False , {}) , ((5, 4) , 2 , 0 , (5, 5) , False , {}) ,
                        ((5, 5) , 2 , 0 , (5, 6) , False , {}) , ((5, 6) , 2 , 0 , (5, 6) , False , {}) ,
                        ((6, 1) , 2 , 0 , (6, 2) , False , {}) , ((6, 2) , 2 , 0 , (6, 3) , False , {}) ,
                        ((6, 3) , 2 , 0 , (6, 4) , False , {}) , ((6, 4) , 2 , 0 , (6, 5) , False , {}) ,
                        ((6, 5) , 2 , 1 , (6, 6) , True , {}) , ((1, 1) , 1 , 0 , (2, 1) , False , {}) ,
                        ((2, 1) , 1 , 0 , (3, 1) , False , {}) , ((3, 1) , 1 , 0 , (4, 1) , False , {}) ,
                        ((4, 1) , 1 , 0 , (5, 1) , False , {}) , ((5, 1) , 1 , 0 , (6, 1) , False , {}) ,
                        ((6, 1) , 1 , 0 , (6, 1) , False , {}) , ((1, 2) , 1 , 0 , (2, 2) , False , {}) ,
                        ((2, 2) , 1 , 0 , (3, 2) , False , {}) , ((3, 2) , 1 , 0 , (4, 2) , False , {}) ,
                        ((4, 2) , 1 , 0 , (5, 2) , False , {}) , ((5, 2) , 1 , 0 , (6, 2) , False , {}) ,
                        ((6, 2) , 1 , 0 , (6, 2) , False , {}) , ((1, 3) , 1 , 0 , (2, 3) , False , {}) ,
                        ((2, 3) , 1 , 0 , (3, 3) , False , {}) , ((3, 3) , 1 , 0 , (4, 3) , False , {}) ,
                        ((4, 3) , 1 , 0 , (5, 3) , False , {}) , ((5, 3) , 1 , 0 , (6, 3) , False , {}) ,
                        ((6, 3) , 1 , 0 , (6, 3) , False , {}) , ((1, 4) , 1 , 0 , (2, 4) , False , {}) ,
                        ((2, 4) , 1 , 0 , (3, 4) , False , {}) , ((3, 4) , 1 , 0 , (4, 4) , False , {}) ,
                        ((4, 4) , 1 , 0 , (5, 4) , False , {}) , ((5, 4) , 1 , 0 , (6, 4) , False , {}) ,
                        ((6, 4) , 1 , 0 , (6, 4) , False , {}) , ((1, 5) , 1 , 0 , (2, 5) , False , {}) ,
                        ((2, 5) , 1 , 0 , (3, 5) , False , {}) , ((3, 5) , 1 , 0 , (4, 5) , False , {}) ,
                        ((4, 5) , 1 , 0 , (5, 5) , False , {}) , ((5, 5) , 1 , 0 , (6, 5) , False , {}) ,
                        ((6, 5) , 1 , 0 , (6, 5) , False , {}) , ((1, 6) , 1 , 0 , (2, 6) , False , {}) ,
                        ((2, 6) , 1 , 0 , (3, 6) , False , {}) , ((3, 6) , 1 , 0 , (4, 6) , False , {}) ,
                        ((4, 6) , 1 , 0 , (5, 6) , False , {}) , ((5, 6) , 1 , 1 , (6, 6) , True , {}) ,
                        ((6, 1) , 3 , 0 , (5, 1) , False , {}) , ((5, 1) , 3 , 0 , (4, 1) , False , {}) ,
                        ((4, 1) , 3 , 0 , (3, 1) , False , {}) , ((3, 1) , 3 , 0 , (2, 1) , False , {}) ,
                        ((2, 1) , 3 , 0 , (1, 1) , False , {}) , ((1, 1) , 3 , 0 , (1, 1) , False , {}) ,
                        ((6, 2) , 3 , 0 , (5, 2) , False , {}) , ((5, 2) , 3 , 0 , (4, 2) , False , {}) ,
                        ((4, 2) , 3 , 0 , (3, 2) , False , {}) , ((3, 2) , 3 , 0 , (2, 2) , False , {}) ,
                        ((2, 2) , 3 , 0 , (1, 2) , False , {}) , ((1, 2) , 3 , 0 , (1, 2) , False , {}) ,
                        ((6, 3) , 3 , 0 , (5, 3) , False , {}) , ((5, 3) , 3 , 0 , (4, 3) , False , {}) ,
                        ((4, 3) , 3 , 0 , (3, 3) , False , {}) , ((3, 3) , 3 , 0 , (2, 3) , False , {}) ,
                        ((2, 3) , 3 , 0 , (1, 3) , False , {}) , ((1, 3) , 3 , 0 , (1, 3) , False , {}) ,
                        ((6, 4) , 3 , 0 , (5, 4) , False , {}) , ((5, 4) , 3 , 0 , (4, 4) , False , {}) ,
                        ((4, 4) , 3 , 0 , (3, 4) , False , {}) , ((3, 4) , 3 , 0 , (2, 4) , False , {}) ,
                        ((2, 4) , 3 , 0 , (1, 4) , False , {}) , ((1, 4) , 3 , 0 , (1, 4) , False , {}) ,
                        ((6, 5) , 3 , 0 , (5, 5) , False , {}) , ((5, 5) , 3 , 0 , (4, 5) , False , {}) ,
                        ((4, 5) , 3 , 0 , (3, 5) , False , {}) , ((3, 5) , 3 , 0 , (2, 5) , False , {}) ,
                        ((2, 5) , 3 , 0 , (1, 5) , False , {}) , ((1, 5) , 3 , 0 , (1, 5) , False , {}) ,
                        ((5, 6) , 0 , 0 , (5, 5) , False , {}) , ((5, 5) , 0 , 0 , (5, 4) , False , {}) ,
                        ((5, 4) , 0 , 0 , (5, 3) , False , {}) , ((5, 3) , 0 , 0 , (5, 2) , False , {}) ,
                        ((5, 2) , 0 , 0 , (5, 1) , False , {}) , ((5, 1) , 0 , 0 , (5, 1) , False , {}) ,
                        ((4, 6) , 0 , 0 , (4, 5) , False , {}) , ((4, 5) , 0 , 0 , (4, 4) , False , {}) ,
                        ((4, 4) , 0 , 0 , (4, 3) , False , {}) , ((4, 3) , 0 , 0 , (4, 2) , False , {}) ,
                        ((4, 2) , 0 , 0 , (4, 1) , False , {}) , ((4, 1) , 0 , 0 , (4, 1) , False , {}) ,
                        ((3, 6) , 0 , 0 , (3, 5) , False , {}) , ((3, 5) , 0 , 0 , (3, 4) , False , {}) ,
                        ((3, 4) , 0 , 0 , (3, 3) , False , {}) , ((3, 3) , 0 , 0 , (3, 2) , False , {}) ,
                        ((3, 2) , 0 , 0 , (3, 1) , False , {}) , ((3, 1) , 0 , 0 , (3, 1) , False , {}) ,
                        ((2, 6) , 0 , 0 , (2, 5) , False , {}) , ((2, 5) , 0 , 0 , (2, 4) , False , {}) ,
                        ((2, 4) , 0 , 0 , (2, 3) , False , {}) , ((2, 3) , 0 , 0 , (2, 2) , False , {}) ,
                        ((2, 2) , 0 , 0 , (2, 1) , False , {}) , ((2, 1) , 0 , 0 , (2, 1) , False , {}) ,
                        ((1, 6) , 0 , 0 , (1, 5) , False , {}) , ((1, 5) , 0 , 0 , (1, 4) , False , {}) ,
                        ((1, 4) , 0 , 0 , (1, 3) , False , {}) , ((1, 3) , 0 , 0 , (1, 2) , False , {}) ,
                        ((1, 2) , 0 , 0 , (1, 1) , False , {}) , ((1, 1) , 0 , 0 , (1, 1) , False , {})]

        self.current_index = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        s = self.Trajectories[self.current_index][0]

        return s

    def step(self, action):

        s, a, r, s_t_1, done, info = self.Trajectories[self.current_index]

        self.current_index += 1

        if self.current_index > len(self.Trajectories):
            self.current_index = 0

        return s_t_1, r, done, info

class ReseedWrapper(gym.core.Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.env.seed(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class NESWActions(gym.core.Wrapper):
    """
    we change the actions to be 0:N, 1:E, 2:S, 3:W
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        if action == 0:
            dir = self.unwrapped.agent_dir
            while dir != 3: # set the correct direction
                self.env.step(0)
                dir = self.unwrapped.agent_dir

        elif action == 1:
            dir = self.unwrapped.agent_dir
            while dir != 0: # set the correct direction
                self.env.step(0)
                dir = self.unwrapped.agent_dir

        elif action == 2:
            dir = self.unwrapped.agent_dir
            while dir != 1: # set the correct direction
                self.env.step(0)
                dir = self.unwrapped.agent_dir

        elif action == 3:
            dir = self.unwrapped.agent_dir
            while dir != 2: # set the correct direction
                self.env.step(0)
                dir = self.unwrapped.agent_dir

        obs, reward, done, info = self.env.step(2) # move forward
        x, y = tuple(self.unwrapped.agent_pos)
        dir = self.unwrapped.agent_dir

        pos_dir = (x, y, dir)

        return pos_dir, reward, done, info
    
    
    def get_goal_position(self):
        return self.goal_position


from .minigrid import Goal


class PositionOnlyObs(gym.core.Wrapper):
    """
    return (x, y) as observation
    """

    def __init__(self, env):
        super().__init__(env)
        self.goal_position = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        env = self.unwrapped
        x, y = tuple(env.agent_pos)

        pos = (x, y)

        return pos, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if not self.goal_position:
            self.goal_position = [x for x, y in enumerate(self.grid.grid) if isinstance(y, (Goal))]
            if len(
                    self.goal_position) >= 1:  # in case there are multiple goals , needs to be handled for other env types
                self.goal_position = (int(self.goal_position[0] / self.height), self.goal_position[0] % self.width)

        env = self.unwrapped
        x, y = tuple(env.agent_pos)

        pos = (x, y)

        return pos

    def get_goal_position(self):
        return self.goal_position

from .minigrid import Goal
class PositionObs(gym.core.Wrapper):
    """
    return (x, y, dir) as observation
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.goal_position = None
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        env = self.unwrapped
        x,y = tuple(env.agent_pos)
        dir = env.agent_dir

        pos_dir = (x, y, dir)

        return pos_dir, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        if not self.goal_position:
            self.goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) ) ]
            if len(self.goal_position) >= 1: # in case there are multiple goals , needs to be handled for other env types
                self.goal_position = (int(self.goal_position[0]/self.height) , self.goal_position[0]%self.width)

        env = self.unwrapped
        x, y = tuple(env.agent_pos)
        dir = env.agent_dir

        pos_dir = (x, y, dir)

        return pos_dir

    def get_goal_position(self):
        return self.goal_position

class KeyDoorTreasureObs(gym.core.Wrapper):
    """
    return (x,y) as observation
    """

    def __init__(self, env):
        super().__init__(env)
        self.key = 0
        self.door = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        x,y = tuple(env.agent_pos)
        dir = env.agent_dir

        if env.carrying is not None and self.key == 0:
            self.key = 1
            reward += 1

        if self.door == 0:
            # Get the position in front of the agent
            fwd_pos = env.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = env.grid.get(*fwd_pos)

            if fwd_cell:
                if hasattr(fwd_cell, "is_open"):
                    if fwd_cell.is_open is True:
                        self.door = 1
                        reward += 1

        inventory = (self.key + self.door)

        pos_dir_inv = (x, y, dir, inventory)

        return pos_dir_inv, reward, done, info

    def reset(self, **kwargs):
        self.key = 0
        self.door = 0
        obs = self.env.reset(**kwargs)

        env = self.unwrapped
        x, y = tuple(env.agent_pos)
        dir = env.agent_dir

        if env.carrying is not None and self.key == 0:
            self.key = 1
            reward += 1

        if self.door == 0:
            # Get the position in front of the agent
            fwd_pos = env.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = env.grid.get(*fwd_pos)

            if fwd_cell:
                if hasattr(fwd_cell, "is_open"):
                    if fwd_cell.is_open is True:
                        self.door = 1
                        reward += 1

        inventory = (self.key + self.door)

        pos_dir_inv = (x, y, dir, inventory)

        return pos_dir_inv

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class OneHotPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype='uint8'
        )

    def observation(self, obs):
        img = obs['image']
        out = np.zeros(self.observation_space.spaces['image'].shape, dtype='uint8')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {
            'mission': obs['mission'],
            'image': out
        }

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }


class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env.get_obs_render(
            obs['image'],
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img_partial
        }

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return {
            'mission': obs['mission'],
            'image': full_grid
        }

class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

class ViewSizeWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

from .minigrid import Goal
class DirectionObsWrapper(gym.core.ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}
    """
    def __init__(self, env,type='slope'):
        super().__init__(env)
        self.goal_position = None
        self.type = type

    def reset(self):
        obs = self.env.reset()
        if not self.goal_position:
            self.goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) ) ]
            if len(self.goal_position) >= 1: # in case there are multiple goals , needs to be handled for other env types
                self.goal_position = (int(self.goal_position[0]/self.height) , self.goal_position[0]%self.width)
        return obs

    def observation(self, obs):
        slope = np.divide( self.goal_position[1] - self.agent_pos[1] ,  self.goal_position[0] - self.agent_pos[0])
        obs['goal_direction'] = np.arctan( slope ) if self.type == 'angle' else slope
        return obs
