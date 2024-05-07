import argparse
import os
import numpy as np

from SimpleNormFlow.simple_nf import SimpleNF
import torch

from linear_evaluation import create_data_loaders_from_arrays
from simclr.modules import SimpleFCClassifier
from utils import yaml_config_hook


def train(args, loader, model, optimizer):
    loss_epoch = 0
    model.train()
    if args.simple_nf_loss == "hybrid":
        classifier = SimpleFCClassifier(512, 10)
        classifier.load_state_dict(torch.load(os.path.join(args.model_path, "classifier_weights.tar"), args.device.type))
        classifier = classifier.to(args.device)
        classifier.eval()
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        if args.simple_nf_loss == "hybrid":
            p_x, p_xcs = model.hybrid_loss_gen_part(x)
            classifier_logits = classifier(x)
            p_cxs = torch.softmax(classifier_logits, dim=1)
            loss_part_1 = torch.sum(torch.tensor([p_cxs[i]*torch.log(p_cxs[i]/0.1) for i in range(10)]))
            loss_part_2 = torch.sum(torch.tensor([p_cxs[i]*torch.log(p_x/p_xcs[i]) for i in range(10)]))

            loss = loss_part_1 + loss_part_2
        else:
            log_px = model.log_prob(x, y)
            loss = - log_px.mean()
        loss_reg = model.gather_regularization()
        total_loss = loss + 0.01 * loss_reg
        total_loss.backward()
        loss_epoch += total_loss
        optim.step()
        if step % 20 == 0:
            print(f"Iter {step + 1}: Loss:{total_loss.item()} Reg:{loss_reg.item()}")

    return loss_epoch

def test(args, loader, model):
    loss_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        log_px = model.log_prob(x, y)
        loss = - log_px.mean()
        loss_reg = model.gather_regularization()
        total_loss = loss + 0.01 * loss_reg

        if step % 20 == 0:
            print(f"Iter {step + 1}: Loss:{total_loss.item()} Reg:{loss_reg.item()}")

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

    print(max(train_X[0]), min(train_X[0]))
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    flow = SimpleNF(512, num_steps=7)
    flow.to(args.device)
    optim = torch.optim.SGD(flow.parameters(), lr=1e-4)

    for epoch in range(args.simple_nf_epochs):
        loss = train(args, arr_train_loader, flow, optim)
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}: Loss: {loss/len(train_y)}")

    test_loss = test(args, arr_test_loader, flow)
    print(f"[Test] Loss: {test_loss/len(test_y)}")

    # Get bits per dim
    n = 0
    bpd_cum = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(arr_test_loader):
            x = x.to(args.device)
            y = y.to(args.device)
            nll = - flow.log_prob(x, y)
            nll_np = nll.cpu().numpy()
            bpd_cum += np.nansum((nll_np / x.shape[1]) / np.log(2))
            n += len(x) - np.sum(np.isnan(nll_np))

        print('Bits per dim: ', bpd_cum / n)
