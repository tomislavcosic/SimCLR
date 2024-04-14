import argparse
import os

from SimpleNormFlow.simple_nf import SimpleNF
import torch

from linear_evaluation import create_data_loaders_from_arrays
from utils import yaml_config_hook


def train(args, loader, model, optimizer):
    loss_epoch = 0
    model.train()
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)

        log_px = model.log_prob(x)
        loss = - log_px.mean()
        loss_reg = model.gather_regularization()
        total_loss = loss + 0.001 * loss_reg
        total_loss.backward()
        loss_epoch += total_loss
        optim.step()
        if step % 20 == 0:
            print(f"Iter {step + 1}: Loss:{total_loss.item()/args.simple_nf_batch_size} Reg:{loss_reg.item()}")

    return loss_epoch

def test(args, loader, model):
    loss_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)

        log_px = model.log_prob(x)
        loss = - log_px.mean()
        loss_reg = model.gather_regularization()
        total_loss = loss + 0.001 * loss_reg
        total_loss.backward()

        if step % 20 == 0:
            print(f"Iter {step + 1}: Loss:{total_loss.item()/args.simple_nf_batch_size} Reg:{loss_reg.item()}")

        loss_epoch += total_loss

    return loss_epoch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_X = torch.load(os.path.join(args.feature_save_path, "train_X.pt"))
    train_y = torch.load(os.path.join(args.feature_save_path, "train_y.pt"))
    test_X = torch.load(os.path.join(args.feature_save_path, "test_X.pt"))
    test_y = torch.load(os.path.join(args.feature_save_path, "test_y.pt"))

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    flow = SimpleNF(512, num_steps=2)
    flow.to(args.device)
    optim = torch.optim.SGD(flow.parameters(), lr=1e-1)

    for epoch in range(args.simple_nf_epochs):
        loss = train(args, arr_train_loader, flow, optim)
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}: Loss: {loss/args.simple_nf_batch_size}")

    test_loss = test(args, arr_test_loader, flow)
    print(f"[Test] Loss: {test_loss}")


