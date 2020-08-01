import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

sys.path.append("../")
import src.resnet50 as resnet50
import src.resnet50atac as resnet50atac


def train_step(x, t, model, optimizer, criterion):
    model.train()
    preds = model(x)
    loss = criterion(preds, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, preds


def test_step(x, t, model, optimizer, criterion):
    model.eval()
    preds = model(x)
    loss = criterion(preds, t)
    return loss, preds


def main(args):
    # Reading Arguments
    enable_atac = args.atac
    if enable_atac:
        print("ATAC Enabled")
    else:
        print("ATAC Disabled")
    epochs = args.epoch
    reduction_ratio = args.reduction

    # Data Preparation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="../data/input", train=True, download=True, transform=transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR10(
        root="../data/input", train=False, download=True, transform=transform
    )
    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=4
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Model Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    if enable_atac:
        model = resnet50atac.ResNet50ATAC(10, r=reduction_ratio).to(device)
    else:
        model = resnet50.ResNet50(10).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)

    # Training Loop
    n_batch_train = len(train_dataloader)
    n_batch_test = len(test_dataloader)
    loss_record = []

    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0
        print(f"Ep. {epoch:>4}:")
        for b_no, (x, t) in enumerate(train_dataloader):
            print(f"[train] batch no. {b_no:>5} / {n_batch_train:>5}", end="\r")
            x, t = x.to(device), t.to(device)
            loss, _ = train_step(x, t, model, optimizer, criterion)
            train_loss += loss.item()
        train_loss /= n_batch_train
        for b_no, (x, t) in enumerate(test_dataloader):
            print(f"[test]  batch no. {b_no:>5} / {n_batch_test:>5}", end="\r")
            x, t = x.to(device), t.to(device)
            loss, preds = test_step(x, t, model, optimizer, criterion)
            test_loss += loss.item()
            test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= n_batch_test
        test_acc /= n_batch_test

        loss_record.append([train_loss, test_loss])
        print(f"Valid Loss: {test_loss:.3f}, Valid Acc: {test_acc:.3f}")
    loss_record = np.array(loss_record)

    # Performance Visualization
    plt.plot(loss_record[:, 0], label="train")
    plt.plot(loss_record[:, 1], label="valid")
    plt.legend()
    plt.grid()
    plt.savefig(f"../plot/{model.__class__.__name__}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--atac", action="store_true", help="Use ATAC")
    parser.add_argument("-e", "--epoch", default=5, type=int, help="# of epoch")
    parser.add_argument(
        "-r", "--reduction", default=0.25, type=float, help="Reduction rate (0~1)"
    )
    args = parser.parse_args()
    main(args)
