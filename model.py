import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class TokenPredictor(nn.Module):
    def __init__(self):
        super(TokenPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # Inputs: concentration, early_buys, social_momentum
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train_model():
    # Load historical data (CSV example; fetch real via APIs)
    data = pd.DataFrame({
        'concentration': [0.5, 0.3, 0.8, 0.4],
        'early_buys': [100, 200, 50, 150],
        'social_momentum': [50, 100, 20, 80],
        'survival_prob': [0.6, 0.9, 0.2, 0.7]  # 1 if hit $1M MC
    })
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
    
    model = TokenPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), 'model.pth')
    return model

def predict(model, features):
    model.eval()
    input_tensor = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        prob = model(input_tensor).item()
    rug_risk = 1 - prob  # Simplified
    reward = prob * 5  # e.g., potential 5x
    return prob, rug_risk, reward

# What-if simulation
def simulate_what_if(model, base_features, change_dict):
    features = base_features.copy()
    for key, multiplier in change_dict.items():
        if key == 'social_momentum':
            features[2] *= multiplier
    return predict(model, features)
