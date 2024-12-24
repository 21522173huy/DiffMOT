import torch
import torch.nn as nn
import torch.nn.functional as F

class FCPositionPredictor(nn.Module):
    """
    Fully connected position predictor. This is baseline model.

    Input: conditions (batch_size, interval, 8), with 8 stands for (x, y, w, h, delta_x, delta_y, delta_w, delta_h)
    Output: delta_bbox (batch_size, 4)
    """
    def __init__(self):
        super(FCPositionPredictor, self).__init__()

        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 8)

    def forward(self, conditions):
        x = torch.relu(self.fc1(conditions))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        delta_bbox = self.fc9(x)
        return delta_bbox

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, t):
        t = t.unsqueeze(-1)  # Add a dimension for the linear layer
        time_emb = self.linear(t)
        return time_emb

class LinearNetwork(nn.Module):
    def __init__(self):
        super(LinearNetwork, self).__init__()
        self.time_embedding = TimeEmbedding(8)
        self.encoder = nn.TransformerEncoderLayer(d_model=8, nhead=1)
        self.predictor = FCPositionPredictor()

    def forward(self, input, beta):
        time_emb = self.time_embedding(beta) # B, 8
        time_emb = time_emb.unsqueeze(1) # B, 1, 8
        concat_input = torch.cat([input, time_emb], dim=1)  # B, (1+ interval + 1), 8
        encoded_input = self.encoder(concat_input) # B, (1 + interval + 1), 8
        desired_input = encoded_input[:, 0, :] # B, 8

        delta_bbox = self.predictor(desired_input)

        return delta_bbox
