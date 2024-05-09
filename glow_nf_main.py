import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import normflows

import os

from utils import yaml_config_hook


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader




class ClassConditionedNormalizingFlow(nn.Module):
    def __init__(self, num_classes, input_size, flow_depth=8, hidden_size=64):
        super(ClassConditionedNormalizingFlow, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.flow_depth = flow_depth
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_classes, hidden_size)
        flows = []
        for _ in range(flow_depth):
            flows.append(normflows.flows.MaskedAffineFlow(input_size + hidden_size))
        self.flow = normflows.NormalizingFlow([normflows.distributions.ClassCondDiagGaussian(512,10)], flows)

    def forward(self, x, y):
        emb = self.embedding(y)
        x = torch.cat([x, emb], dim=1)  # concatenate input with class embedding
        return self.flow(x)


def bits_per_dimension(loss, num_dims):
    return loss / (num_dims * torch.log(torch.tensor(2.0)))


# Example usage:
num_classes = 10
input_size = 128  # Adjust this according to your feature size
flow_depth = 8
hidden_size = 64
model = ClassConditionedNormalizingFlow(num_classes, input_size, flow_depth, hidden_size)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define loss function (you might need to define your own based on your task)
criterion = nn.MSELoss()


# Example training loop
def train(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, inputs)  # Example loss calculation
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")



parser = argparse.ArgumentParser(description="SimCLR")
config = yaml_config_hook("./config/config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
args = parser.parse_args()

train_X = torch.load(os.path.join(args.feature_save_path, "train_X.pt"))
train_y = torch.load(os.path.join(args.feature_save_path, "train_y.pt"))
test_X = torch.load(os.path.join(args.feature_save_path, "test_X.pt"))
test_y = torch.load(os.path.join(args.feature_save_path, "test_y.pt"))

train_loader, test_loader = create_data_loaders_from_arrays(
    train_X, train_y, test_X, test_y, args.logistic_batch_size
)
# Example usage of the training loop
num_epochs = 10
train(model, train_loader, optimizer, criterion, num_epochs)


# Example usage of the model for inference
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_dims = 0
        for inputs, labels in test_loader:
            outputs = model(inputs, labels)
            loss = criterion(outputs, inputs)
            total_loss += loss.item() * inputs.size(0)
            total_dims += inputs.size(0) * inputs.size(1)  # Number of elements in inputs

        avg_loss = total_loss / len(test_loader.dataset)
        bpd = bits_per_dimension(avg_loss, total_dims)
        print(f"Average Test Loss: {avg_loss:.4f}, Bits per Dimension: {bpd:.4f}")


# Example usage of the test function
test(model, test_loader)
