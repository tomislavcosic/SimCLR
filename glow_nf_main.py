import argparse
import os

import torch
import numpy as np
import normflows as nf

from tqdm import tqdm

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


if __name__ == '__main__':

    # Set up model

    # Define flows
    L = 3
    K = 16
    torch.manual_seed(0)

    input_shape = [512]
    n_dims = np.prod(input_shape)
    channels = 1
    hidden_channels = 256
    split_mode = 'checkerboard'
    scale = True
    num_classes = 10

    # Set up flows, distributions and merge operations
    q0 = nf.distributions.ClassCondDiagGaussian(512, num_classes)
    flows = []
    for i in range(L):
        flows.append(nf.flows.InvertibleAffine(512))


    # Construct flow model with the multiscale architecture
    model = nf.ClassCondFlow(q0, flows)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)


    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    train_X = torch.load(os.path.join(args.feature_save_path, "train_X.pt"))
    train_y = torch.load(os.path.join(args.feature_save_path, "train_y.pt"))
    test_X = torch.load(os.path.join(args.feature_save_path, "test_X.pt"))
    test_y = torch.load(os.path.join(args.feature_save_path, "test_y.pt"))

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    train_iter = iter(arr_train_loader)

    # Train model
    max_iter = 2000

    loss_hist = np.array([])

    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for i in tqdm(range(max_iter)):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(arr_train_loader)
            x, y = next(train_iter)
        optimizer.zero_grad()
        loss = -model.log_prob(x.to(device), y.to(device))

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())

    # Get bits per dim
    n = 0
    bpd_cum = 0
    with torch.no_grad():
        for x, y in iter(arr_test_loader):
            nll = -model.log_prob(x.to(device), y.to(device))
            nll_np = nll.cpu().numpy()
            bpd_cum += np.nansum(nll_np / np.log(2) / n_dims)
            n += len(x) - np.sum(np.isnan(nll_np))

        print('Bits per dim: ', bpd_cum / n)
