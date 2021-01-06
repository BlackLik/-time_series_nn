import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self, n_inputs=32):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_inputs*2)
        self.fc2 = nn.Linear(n_inputs*2, n_inputs)
        self.fc3 = nn.Linear(n_inputs, n_inputs//2)
        self.fc4 = nn.Linear(n_inputs//2, 1)
        self.act1 = nn.ReLU()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class NetRegressor(object):
    def __init__(self, n_input=32, n_epochs=100, batch_size=64, lr=0.001, lam=0.001,  optimizer='RMSprop', debug=0):
        self.model = None
        self.n_input = n_input
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer
        self.debug = debug
        self.ss = StandardScaler()
    
    def fit(self, X, y):
        #X = self.ss.fit_transform(X)
        # Estimate model
        self.model = Net(n_inputs=self.n_input).to(device)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.MSELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Enable droout
        self.model.train(True)
        # Start the model fit
        
        best_loss = float('inf')
        best_state = None
        
        for epoch_i in range(self.n_epochs):
            loss_history = []
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                y_pred_batch = self.model(x_batch)
                loss = loss_func(y_batch, y_pred_batch)
                
                lam = torch.tensor(self.lam)
                l2_reg = torch.tensor(0.).to(device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += lam * l2_reg
                
                # set gradients to zero
                opt.zero_grad()
                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()
                loss_history.append(loss.item())
            if self.debug:
                print("epoch: %i, mean loss: %.5f" % (epoch_i, np.mean(loss_history)))
            
        
            best_loss = np.mean(loss_history)
            best_state = self.model.state_dict()
        
        self.model.load_state_dict(best_state)

    def predict(self, X):
        #X = self.ss.transform(np.array(X).reshape(-1, self.n_input))
        # Disable droout
        self.model.train(False)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        # Make predictions for X 
        y_pred = self.model(X_tensor)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.minimum(y_pred, 0.05)
        return np.maximum(y_pred, 0)
        
class NetAugmentation(object):
    def __init__(self):
        self.reg      = None
        self.y_scaler = None
        self.X_scaler = None
    
    def fit(self, price, date):
        
        price = np.array(price)
        date = np.array(date)
        
        self.X_scaler = StandardScaler().fit(date.reshape((-1, 1)))
        X_ss = self.X_scaler.transform(date.reshape((-1, 1)))
        self.y_scaler = StandardScaler().fit(price.reshape((-1, 1)))
        y_ss = self.y_scaler.transform(price.reshape((-1, 1)))
        
        self.reg = NetRegressor(n_epochs=10, batch_size=64, lr=0.01, lam=1.,  optimizer='RMSprop', debug=0)
        self.reg.fit(X_ss, y_ss)
    
    def predict(self, date):
        date = np.array(date)
        X_ss = self.X_scaler.transform(date.reshape((-1, 1)))
        price_pred = self.y_scaler.inverse_transform(self.reg.predict(X_ss))
        
        return price_pred