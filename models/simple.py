import torch
import torch.nn as nn
import torch.nn.functional as F

class FCPositionPredictor(nn.Module):
    """
    Fully connected position predictor. This is baseline model.

    Input: conditions (batch_size, interval, 8), with 8 stands for (x, y, w, h, delta_x, delta_y, delta_w, delta_h)
    Output: delta_bbox (batch_size, 4)
    """
    def __init__(self, in_features, out_features):
        super(FCPositionPredictor, self).__init__()

        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, out_features)

    def forward(self, conditions):
        x = torch.relu(self.fc1(conditions))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        delta_bbox = self.fc9(x)
        return delta_bbox

class LinearNetwork(nn.Module):
    def __init__(self):
        super(LinearNetwork, self).__init__()
        self.predictor = FCPositionPredictor(in_features = 4 + 256 + 3,
                                             out_features=4)

    def forward(self, x, beta, context):
        batch_size = x.size(0) # B, 8
        beta = beta.view(batch_size, 1) # (B, 1)
        context = context.view(batch_size, -1)   # (B, 64)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)

        concat_input = torch.cat([x, time_emb, context], dim=-1)    # (B, 8 + F + 3)
        delta_bbox = self.predictor(concat_input)

        return delta_bbox