import os
import random
import pickle
import numpy as np
from operator import itemgetter

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.position = 0

        self.global_labels_pc_buffer = []
        self.push_actions_buffer = []
        self.actions_idx_buffer = []
        self.next_global_labels_pc_buffer = []
        self.next_push_actions_buffer = []
        self.reward_buffer = []
        self.done_buffer = []

    def push(self, global_labels_pc, push_actions, actions_idx, reward, next_global_labels_pc, next_push_actions, mask_done):

        if len(self.global_labels_pc_buffer) < self.capacity:
            self.global_labels_pc_buffer.append(None)
            self.push_actions_buffer.append(None)
            self.actions_idx_buffer.append(None)
            self.next_global_labels_pc_buffer.append(None)
            self.next_push_actions_buffer.append(None)  
            self.reward_buffer.append(None)
            self.done_buffer.append(None)    
        # !!! newaxis for batch size 1 !!!
        self.global_labels_pc_buffer[self.position] = global_labels_pc        
        self.push_actions_buffer[self.position] = push_actions
        self.actions_idx_buffer[self.position] = actions_idx
        self.next_global_labels_pc_buffer[self.position] = next_global_labels_pc
        self.next_push_actions_buffer[self.position] = next_push_actions
        self.reward_buffer[self.position] = np.array([reward])
        self.done_buffer[self.position] = np.array([mask_done])

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(range(len(self.global_labels_pc_buffer)), batch_size) 
        global_pc_batch = itemgetter(*batch)(self.global_labels_pc_buffer)
        push_actions_batch = itemgetter(*batch)(self.push_actions_buffer)
        actions_idx_batch = itemgetter(*batch)(self.actions_idx_buffer)
        next_global_pc_batch = itemgetter(*batch)(self.next_global_labels_pc_buffer)
        next_push_actions_batch = itemgetter(*batch)(self.next_push_actions_buffer)
        reward_batch = itemgetter(*batch)(self.reward_buffer)
        done_batch = itemgetter(*batch)(self.done_buffer)
        
        return global_pc_batch, push_actions_batch, actions_idx_batch, reward_batch, next_global_pc_batch, next_push_actions_batch, done_batch

    def __len__(self):
        return len(self.global_labels_pc_buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
