import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

from agent import Agent
from model import Actor, Critic

BUFFER_SIZE = int(3e5)  # replay buffer size
BATCH_SIZE = 256       # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG():
    
    def __init__(self, ob_size, action_size, random_seed):
        """Initialize 2 Agent objects.
        
        Params
        ======
            ob_size (int): dimension of one agent's observation
            action_size (int): dimension of each action
        """
        self.ob_size = ob_size
        self.action_size = action_size        
        # Initialize the agents
        self.agent1 = Agent(ob_size, action_size, random_seed = 0)
        self.agent2 = Agent(ob_size, action_size, random_seed = 1)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Time steps for UPDATE EVERY
        self.t_step = 0
    
    def act(self, states, rand = False):
        """Agents act with actor_local"""
        if rand == False:
            action1 = self.agent1.act(states[0])
            action2 = self.agent2.act(states[1])
            actions = [action1, action2]
            return actions
        if rand == True:
            actions = np.random.randn(2, 2) 
            actions = np.clip(actions, -1, 1)
            return actions
 
        
    def step(self, states, actions, rewards, next_states, dones, learn = True):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        state1 = states[0]
        state2 = states[1]
        
        action1 = actions[0]
        action2 = actions[1]
        
        reward1 = rewards[0]
        reward2 = rewards[1]
        
        next_state1 = next_states[0]
        next_state2 = next_states[1]
        
        done1 = dones[0]
        done2 = dones[1]

        self.memory.add(state1, state2, action1, action2, reward1, reward2, next_state1, next_state2, done1, done2)
        
        self.t_step += 1

        # Learn, if enough samples are available in memory
        if self.t_step % UPDATE_EVERY == 0:
            if learn == True and len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
     
    def learn(self, experiences, GAMMA):
        states1, states2, actions1, actions2, rewards1, rewards2, next_states1, next_states2, dones1, dones2 = experiences
        
        # next actions (for CRITIC network)
        actions_next1 = self.agent1.actor_target(next_states1)
        actions_next2 = self.agent2.actor_target(next_states2)
        
        # action predictions (for ACTOR network)
        actions_pred1 = self.agent1.actor_local(states1)
        actions_pred2 = self.agent2.actor_local(states2)
        
        # m_obs, o_obs, m_actions, o_actions, m_next_obs, o_next_obs, m_rewards, o_rewards, m_ds, o_ds, m_na, o_na
        self.agent1.learn(states1, states2, actions1, actions2, rewards1, rewards2, next_states1, next_states2, dones1, dones2, actions_next1, actions_next2, actions_pred1, actions_pred2)
        self.agent2.learn(states2, states1, actions2, actions1, rewards2, rewards1, next_states2, next_states1, dones2, dones1, actions_next2, actions_next1, actions_pred2, actions_pred1)
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state1","state2", 
                                                                "action1", "action2",
                                                                "reward1", "reward2",
                                                                "next_state1", "next_state2",
                                                                "done1", "done2"])
        self.seed = random.seed(seed)
    
    def add(self, state1, state2, action1, action2, reward1, reward2, next_state1, next_state2, done1, done2):
        """Add a new experience to memory."""
        e = self.experience(state1, state2, action1, action2, reward1, reward2, next_state1, next_state2, done1, done2)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states1 = torch.from_numpy(np.vstack([e.state1 for e in experiences if e is not None])).float().to(device)
        states2 = torch.from_numpy(np.vstack([e.state2 for e in experiences if e is not None])).float().to(device)
        
        actions1 = torch.from_numpy(np.vstack([e.action1 for e in experiences if e is not None])).float().to(device)
        actions2 = torch.from_numpy(np.vstack([e.action2 for e in experiences if e is not None])).float().to(device)
        
        rewards1 = torch.from_numpy(np.vstack([e.reward1 for e in experiences if e is not None])).float().to(device)
        rewards2 = torch.from_numpy(np.vstack([e.reward2 for e in experiences if e is not None])).float().to(device)
        
        next_states1 = torch.from_numpy(np.vstack([e.next_state1 for e in experiences if e is not None])).float().to(device)
        next_states2 = torch.from_numpy(np.vstack([e.next_state2 for e in experiences if e is not None])).float().to(device)
        
        dones1 = torch.from_numpy(np.vstack([e.done1 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        dones2 = torch.from_numpy(np.vstack([e.done2 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states1, states2, actions1, actions2, rewards1, rewards2, next_states1, next_states2, dones1, dones2)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)        
        
        
        