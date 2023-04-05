import numpy as np
import torch
import pandas as pd
import tensorflow as tf
def function():
    data = pd.read_csv(r"C:/Users/suhas/Downloads/Bank_Personal_Loan_Modelling.csv")
    data.drop(['ID'] , axis = 1 , inplace = True)
    x = data.drop(['Personal Loan'] , axis = 1).values
    y = data['Personal Loan'].values
    x = torch.tensor(x , dtype = torch.float64)
    y = torch.tensor(y , dtype=  torch.float64)
    y = y.to(torch.float64)
    from sklearn.model_selection import train_test_split
    x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 42 , test_size = 0.25)
    return x_train , x_test , y_train , y_test

class NN(torch.nn.Module):-
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(12, 10 , bias = False)
        self.linear2 = torch.nn.Linear(10, 20 , bias = False)
        self.linear3 = torch.nn.Linear(20 , 1 , bias = False)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x.float())
        x = self.relu(x.float())
        x = self.linear2(x.float())
        x = self.linear3(x.float())
        x = self.relu(x.float())
        x = self.sigmoid(x.float())
        return x

model = NN()
loss_function = torch.nn.MSELoss()

import torch
import numpy as np
class AntColonyOptimization:
    def __init__(self, nn, ants, iterations, alpha, beta, rho, Q):
        self.nn = nn
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.pheromone = np.ones((self.nn.input_size, self.nn.hidden_size, self.nn.output_size))
        
    def update_pheromone(self, ant, delta_weights):
        # Update the pheromone matrix
        self.pheromone += self.Q * delta_weights * ant
        
    def run(self, X, y):
        for i in range(self.iterations):
            # Reset the ants
            ants = np.zeros((self.ants, self.nn.input_size, self.nn.hidden_size, self.nn.output_size))
            
            for j in range(self.ants):
                # Compute the forward pass through the neural network
                nn_output = self.nn.forward(X)
                
                # Compute the error and the delta weights
                error = y - nn_output
                delta_weights = self.alpha * X.reshape(-1, 1, self.nn.input_size) * self.beta * self.nn.z2.reshape(-1, self.nn.hidden_size, 1) * error.reshape(-1, 1, self.nn.output_size)
                
                # Update the pheromone matrix
                self.update_pheromone(ants[j], delta_weights)
                
                # Update the weights
                self.nn.weights1 += self.rho * delta_weights.sum(axis=3).sum(axis=1)
                self.nn.weights2 += self.rho * self.beta * self.nn.z2.reshape(-1, self.nn.hidden_size, 1) * error.reshape(-1, 1, self.nn.output_size)

        # Return the updated neural network
        return self.nn


        
        

        

























