import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

class TokenPredictor(nn.Module):
    def __init__(self):
        super(TokenPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_model():
    # Dummy training data - replace with real historical data later
    data = pd.DataFrame({
        'concentration': [0.7, 0.4, 0.9, 0.3, 0.6, 0.2],
        'early_buys': [50, 300, 20, 500, 150, 80],
        'social_momentum': [10, 200, 5, 400, 100, 30],
        'survival_prob': [0.1, 0.95, 0.05, 0.99, 0.7, 0.2]
    })
    
    X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
    
    model = TokenPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    if not os.path.exists('model.pth'):
        torch.save(model.state_dict(), 'model.pth')
    
    return model

def predict(model, features):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([features], dtype=torch.float32)
        prob = model(input_tensor).item()
    rug_risk = 1 - prob
    reward = prob * 6  # Scaled potential reward
    return prob, rug_risk, reward

def simulate_what_if(model, base_features, change_dict):
    features = base_features.copy()
    for key, val in change_dict.items():
        if key == 'social_momentum' and len(features) > 2:
            features[2] *= val
    return predict(model, features)
