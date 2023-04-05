import numpy as np
import torch

import pandas as pd
import tensorflow as tf
def function():
    data = pd.read_csv("C:/Users/suhas/Downloads/Bank_Personal_Loan_Modelling.csv")
    data.head()
data.info()
#pytorch
data.drop(['ID'],inplace=True,axis=1)
x=data.drop(['Personal Loan'],axis=1).values
y=data['Personal Loan'].values
x = torch.tensor(x , dtype = torch.float64)
y = torch.tensor(y)
y = y.type(torch.LongTensor)
y
class NN(nn.Module):
    def __init__(self,input_size,hidden1,hidden2,output_size):
        super().__init__()
        self.input_=nn.Linear(input_size,hidden1)
        self.hidden1_=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,output_size)
    def forward(self,x):
        x=f.relu(self.input_(x))
        x=torch.sigmoid(self.hidden1_(x))
        x=self.out(x)
        x=f.sigmoid(x)
        return x
    x.shape
model = NN(12 , 10 , 8 , 4)

model.parameters
import tensorflow as tf
from tensorflow import keras
loss_function = nn.CrossEntropyLoss()
loss_function
import torch.nn.functional as f
import torch.optim
from torch.utils.data import DataLoader , TensorDataset
import tensorflow as tf
from   torch.optim.lr_scheduler import ExponentialLR as ExponentialLR
#from torch.optim.lr_scheduler import ExponentialLR
optimizer = torch.optim.Adam( model.parameters(), lr=100)
scheduler = ExponentialLR(optimizer, gamma=0.9)
n_epochs=1000
data = TensorDataset(x , y)
data = DataLoader(data )
with tf.device('/gpu:0'):
    final_losses = []
    for i in range(n_epochs):
        y_pred = model(x.float())
        loss = loss_function(y_pred , y)
        final_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if (i%10 == 0):
            print("Epoch {} has loss of {}".format(i , loss.item()))


import matplotlib.pyplot as plt
plt.plot(range(1 , n_epochs+1) , final_losses)
from sklearn.metrics import classification_report
print(classification_report(y , predictions))
  

model = NN()
loss_function = torch.nn.MSELoss()

class GeneticOptimizer:
    def __init__(self, model, population_size, mutation , decay ,  inputs  , labels):
        self.model = model
        self.population_size = population_size
        self.mutation = mutation
        self.population = self.init_population()
        self.decay = decay
        self.inputs = inputs
        self.labels = labels

    def init_population(self):
        population = []
        for i in range(self.population_size):
            weights = []
            for weight in self.model.parameters():
                weights.append(weight.data.numpy())
            population.append(weights)
        return population

    def selection(self, fitness_scores):
        cumulative_scores = np.cumsum(fitness_scores)
        total_score = np.sum(fitness_scores)
        rand = np.random.uniform(0, total_score)
        selected_index = np.searchsorted(cumulative_scores, rand)
        return selected_index

    def crossover(self, male, female):
        random_crossover = np.random.randint(1, len(male))
        child1 = male[:random_crossover] + female[random_crossover:]
        child2 = male[:random_crossover] + female[random_crossover:]
        return child1, child2
    
    def decay_mutation_rate(self):
        self.mutation -= (self.decay*self.mutation)

    def mutate(self, child):
        for i in range(len(child)):
            if np.random.uniform(0, 1) < self.mutation:
                child[i] += np.random.normal(0, 0.1, child[i].shape)
        return child

    def generate_offspring(self, fitness_scores):
        new_population = []
        for _ in range(self.population_size):
            parent1_index = self.selection(fitness_scores)
            parent2_index = self.selection(fitness_scores)
            parent1 = self.population[parent1_index]
            parent2 = self.population[parent2_index]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        self.population = new_population

    def update_weight(self):
        fitness_scores = [self.fitness(weights) for weights in self.population]
        best_index = np.argmax(fitness_scores)
        best_weights = self.population[best_index]
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(best_weights[i])

    def fitness(self, weights):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(weights[i])
        outputs = self.model(self.inputs)
        loss = loss_function(outputs.float(), self.labels.reshape([len(self.inputs) , 1]).float())
        return 1 / (loss.item() + 1e-6)

x_train , x_test , y_train , y_test = function()
genetic_optimizer = GeneticOptimizer(model, population_size=20, mutation=0.3  , decay = 0.05 , inputs = x_train, labels = y_train)

def train(num_epochs):
    loss_list = []
    with tf.device('/gpu:0'):
        for epoch in range(num_epochs):
            genetic_optimizer.generate_offspring([])
            genetic_optimizer.update_weight()
            outputs = model(x_train)
            loss = loss_function(outputs, y_train.reshape([len(x_train) , 1]).float())
            loss_list.append(loss.item())
            loss.backward()
            genetic_optimizer.generate_offspring([])
            genetic_optimizer.update_weight()
            if (epoch%10 == 0):
                print("Epoch" , epoch , " : " , loss.item());
                genetic_optimizer.decay_mutation_rate()
    return loss_list
    


