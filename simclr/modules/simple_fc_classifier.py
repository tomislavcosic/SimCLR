import torch.nn as nn


class SimpleFCClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super(SimpleFCClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
