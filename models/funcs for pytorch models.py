import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test(model, loader, device):

    loss_log = []
    acc_log = []
    model.eval()
    loss_func = nn.CrossEntropyLoss()

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            out = model(data)
        loss = loss_func(out, target)

        loss_log.append(loss.item())

        pred = torch.argmax(out, dim=1)
        acc_log.append((pred == target).detach().cpu().numpy().sum() / len(pred))

    return np.mean(loss_log), np.mean(acc_log)


def train_epoch(model, optimizer, train_loader, device):
    loss_log = []
    acc_log = []
    model.train()
    loss_func = nn.CrossEntropyLoss()

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        out = model(data)
        loss = loss_func(out, target)
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())

        pred = torch.argmax(out, dim=1)
        acc_log.append((pred == target).detach().cpu().numpy().sum() / len(pred))

    return loss_log, acc_log


def train(model, optimizer, n_epochs, train_loader, val_loader, scheduler=None):
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []
    train_len_epoch_loss = []
    train_len_epoch_acc = []

    for epoch in range(n_epochs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)
        val_loss, val_acc = test(model, val_loader, device)

        train_loss_log.extend(train_loss)
        train_acc_log.extend(train_acc)
        train_len_epoch_loss.append(len(train_loss_log))
        train_len_epoch_acc.append(len(train_acc_log))

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        # wandb
        run_train = wandb.init(project='deep_learning_hw2', name="train data", resume=(epoch > 0), reinit=True)
        data = [[x, y] for (x, y) in zip(np.linspace(0, len(train_loss_log), len(train_loss_log), dtype='int'), train_loss_log)]
        table = wandb.Table(data=data, columns=["step", "data"])
        run_train.log({"loss": wandb.plot.line(table, "step", "data", stroke=None, title="loss")})

        data = [[x, y] for (x, y) in zip(np.linspace(0, len(train_acc_log), len(train_acc_log), dtype='int'), train_acc_log)]
        table = wandb.Table(data=data, columns=["step", "data"])
        run_train.log({"accuracy": wandb.plot.line(table, "step", "data", stroke=None, title="accuracy")})

        run_val = wandb.init(project='deep_learning_hw2', name="val data", resume=(epoch > 0), reinit=True)
        data = [[x, y] for (x, y) in zip(np.linspace(len(train_loss_log) / (epoch + 1), len(train_loss_log), epoch + 1, dtype='int'), val_loss_log)]
        table = wandb.Table(data=data, columns=["step", "data"])
        run_val.log({"loss": wandb.plot.line(table, "step", "data", stroke=None, title="loss")})

        data = [[x, y] for (x, y) in zip(np.linspace(len(train_acc_log) / (epoch + 1), len(train_acc_log), epoch + 1, dtype='int'), val_acc_log)]
        table = wandb.Table(data=data, columns=["step", "data"])
        run_val.log({"accuracy": wandb.plot.line(table, "step", "data", stroke=None, title="accuracy")})

        print(f"Epoch {epoch}")
        print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
        print(f" val loss: {val_loss}, val acc: {val_acc}\n")

        if scheduler is not None:
            scheduler.step()

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log