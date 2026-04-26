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


class ChaturajiEnv(AECEnv):

    def __init__(self):
        #super().__init__()


        self.embedding_dim = 5
        self.board_embedding_shape = (self.embedding_dim**2, 8, 8)
    
        self.possible_agents = ["blue","red", "green", "yellow"]
        self._action_spaces = {"agent": Discrete(64) for agent in self.possible_agents}
        self._observation_space = Dict({"board": Box(low=0, high=1, shape=self.board_embedding_shape, dtype=np.float32),
                                        "score": Box(low=0 ,high =1, shape = (4,),dtype=np.float32)})
    
    def _add_to_board(self, piece, player,x,y):

        channel = piece * len(self.possible_agents) + player
        self.board_embedding[channel,x,y] = 1
        
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
        self.state[self.agent_selection] = action

        return



env = ChaturajiEnv()
env.reset()




            

        

            


        


        





