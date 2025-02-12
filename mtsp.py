import numpy as np
import matplotlib.pyplot as plt
from typing import List
import torch

class mtsp():
    def __init__(self, n_cities, n_agents, batch_size,train_times, test_times,seed=None):
        assert n_cities > n_agents
        self.n_cities = n_cities
        self.n_agents = n_agents
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.feature_dim = 2
        self.map_length = 1
        self.batch_size = batch_size
        self.train_times = train_times
        self.test_times = test_times
        cities = self.rng.uniform(0, self.map_length, (batch_size*train_times,n_cities, self.feature_dim))
        start = self.rng.uniform(0, self.map_length, (batch_size*train_times,n_agents, self.feature_dim))
        self.X = torch.cat((torch.tensor(start,dtype=torch.float),torch.tensor(cities,dtype=torch.float)),dim=1)
        cities = self.rng.uniform(0, self.map_length, (batch_size*test_times,n_cities, self.feature_dim))
        start = self.rng.uniform(0, self.map_length, (batch_size*test_times,n_agents, self.feature_dim))
        self.tX = torch.cat((torch.tensor(start,dtype=torch.float),torch.tensor(cities,dtype=torch.float)),dim=1)

        # self.distances = np.zeros((n_cities, n_cities))
        # for i in range(n_cities):
        #     for j in range(n_cities):
        #         self.distances[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])

        # self.tour: List[List[int]] = []
        # self.costs = np.zeros(n_agents)
        # self.all_cost = 0

    # def update_tour(self, task_schedules: List[List[int]]):

    def get_batch(self, time_idx, is_train):
        if is_train:
            return self.X[time_idx*self.batch_size:(time_idx+1)*self.batch_size]
        else:
            return self.tX[time_idx*self.batch_size:(time_idx+1)*self.batch_size]
        
    def get_distance(self, seq, sample_idx):
        start = self.tX[sample_idx, 0:self.n_agents]
        cities = self.tX[sample_idx, self.n_agents:]
        dis = 0
        for i in range(self.n_agents):
            dis += np.linalg.norm(start[i] - cities[seq[i][0]])
            for j in range(len(seq[i])-1):
                dis += np.linalg.norm(cities[seq[i][j]] - cities[seq[i][j+1]])
        return dis
        
    #shape of seq: (n_agents, n_cities)
    def render(self,seq,sample_idx):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(0, self.map_length)
        ax.set_ylim(0, self.map_length)
        start = self.tX[sample_idx, 0:self.n_agents]
        cities = self.tX[sample_idx, self.n_agents:]
        plt.scatter(cities[:,0], cities[:,1], s=40, c='blue', marker='o')
        plt.scatter(start[:,0], start[:,1], s=100, c='red')
        for i in range(self.n_cities):
            plt.text(cities[i,0], cities[i,1], str(i))
        
        for i in range(self.n_agents):
            plt.plot([start[i,0], cities[seq[i][0],0]], [start[i,1], cities[seq[i][0],1]], c='black')
            for j in range(len(seq[i])-1):
                plt.plot([cities[seq[i][j],0], cities[seq[i][j+1],0]], 
                         [cities[seq[i][j],1], cities[seq[i][j+1],1]], c='black')
        # plt.scatter(self.cities[:,0], self.cities[:,1], s=40, c='blue', marker='o')
        # plt.scatter(self.start[:,0], self.start[:,1], s=100, c='red')
        # for i in range(self.n_cities):
        #     plt.text(self.cities[i,0], self.cities[i,1], str(i))
        # for i in range(self.n_agents):
        #     plt.text(self.start[i,0], self.start[i,1], str(i))
        # for i in range(self.n_agents):
        #     plt.plot([self.start[i,0], self.cities[self.tour[i][0],0]], [self.start[i,1], self.cities[self.tour[i][0],1]], c='black')
        #     for j in range(len(self.tour[i])-1):
        #         plt.plot([self.cities[self.tour[i][j],0], self.cities[self.tour[i][j+1],0]], 
        #                  [self.cities[self.tour[i][j],1], self.cities[self.tour[i][j+1],1]], c='black')
        plt.show()


if __name__ == '__main__':
    env = mtsp(10, 3)
    env.render()