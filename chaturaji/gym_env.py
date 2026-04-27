import functools
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

reward_dict = {"pawn":1,
               "boat":5,
               "bishop":5,
               "knigth":3,
               "king":3
               }

pawn_moves = {"red": [(-1,0), (-1,1), (-1,-1)],
              "blue": [(0,-1), (1,-1), (-1,-1)],
              "yellow": [(1,0), (1,1), (1,-1)],
              "green": [(0,1), (1,1), (-1,1)]
              }
from itertools import product
ray = [i for i in range(-7,8) if i != 0]

rook_moves = {"red": list(product(ray, [0])) + list(product([0], ray)),
              "blue": list(product(ray, [0])) + list(product([0], ray)),
              "yellow": list(product(ray, [0])) + list(product([0], ray)),
              "green": list(product(ray, [0])) + list(product([0], ray))
              }


knight_moves = {"red": [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)],
                "blue": [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)],
                "yellow": [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)],
                "green": [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
                }

boat_moves = {"red": [(-1,-1), (-1,1), (1,-1), (1,1)],
              "blue": [(-1,-1), (-1,1), (1,-1), (1,1)],
              "yellow": [(-1,-1), (-1,1), (1,-1), (1,1)],
              "green": [(-1,-1), (-1,1), (1,-1), (1,1)]
              }


class ChaturajiEnv(AECEnv):

    def __init__(self):
        self.embedding_dim = 5
        self.board_embedding_shape = (self.embedding_dim**2, 8, 8)
    
        self.possible_agents = ["blue","red", "green", "yellow"]
        self._action_spaces = {"agent": Discrete(64) for agent in self.possible_agents}
        self._observation_space = Dict({"board": Box(low=0, high=1, shape=self.board_embedding_shape, dtype=np.float32),
                                        "score": Box(low=0 ,high =1, shape = (4,),dtype=np.float32)})
    
    def _add_to_board(self, piece, player,x,y):

        channel = piece * len(self.possible_agents) + player
        self.board_embedding[channel,x,y] = 1

    def _remove_from_board(self, piece, player,x,y):

        channel = piece * len(self.possible_agents) + player
        self.board_embedding[channel,x,y] = 0

    def _move_and_kill(self, piece, player,x,y):

        channel = piece * len(self.possible_agents) + player
        self.board_embedding[channel,x,y] = 0

        for p in range(self.embedding_dim):
            for pl in range(len(self.possible_agents)):
                if self.board_embedding[p*len(self.possible_agents)+pl,x,y] == 1:
                    self.board_embedding[p*len(self.possible_agents)+pl,x,y] = 0
                    return reward_dict[p]
        
        return 0

    def _empty_check(self, piece, player,x,y):
        channel_start = player * self.embedding_dim
        channel_end = channel_start+ 4 
        if sum(self.board_embedding[channel_start:channel_end,x,y]) == 0:
            return True
        else:
            return False


    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents

        #Env stuff
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0

        #Chaturaji stuff
        self._next_to_go = 0

        self.board_embedding = np.zeros(self.board_embedding_shape, dtype=np.float32)
        self.scores = np.zeros(4, dtype=np.float32)
        
    
        # Red              
        for i in range(4):
            self._add_to_board(0,0,6,i)
            self._add_to_board(1+i,0,7,i)

        # Green           
        for i in range(4):
            self._add_to_board(0,0,4+i,6)
            self._add_to_board(1+i,0,4+i,7)

        #Yellow      
        for i in range(4):
            self._add_to_board(0,0,1,4+i)
            self._add_to_board(1+i,0,0,4+i)

        #Blue       
        for i in range(4):
            self._add_to_board(0,0,i,1)
            self._add_to_board(1+i,0,i,0)


        board_2d = np.sum(self.board_embedding, axis=0)
        print(board_2d)

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.agent_selection
    
    def step(self, action):


        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return 

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        return


env = ChaturajiEnv()
env.reset()




            

        

            


        


        





