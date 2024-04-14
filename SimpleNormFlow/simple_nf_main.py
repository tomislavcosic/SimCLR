import argparse
import os

from simple_nf import SimpleNF
import torch
import torchvision

from linear_evaluation import get_features, create_data_loaders_from_arrays
from simclr import SimCLR
from simclr.modules import get_resnet
from simclr.modules.transformations import TransformsSimCLR
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

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.simple_nf_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.simple_nf_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    (train_X, train_y, test_X, test_y) = get_features(simclr_model, train_loader, test_loader, args.device)

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    flow = SimpleNF(2, num_steps=7)
    optim = torch.optim.SGD(flow.parameters(), lr=1e-1)

    for epoch in range(args.simple_nf_epochs):
        loss = train(args, arr_train_loader, flow, optim)
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}: Loss: {loss/args.simple_nf_batch_size}")

    test_loss = test(args, arr_test_loader, flow)
    print(f"[Test] Loss: {test_loss}")


