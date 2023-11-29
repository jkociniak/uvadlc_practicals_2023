################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set, set_dataset


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights, progress=False)

    # Randomly initialize and modify the model's last layer for CIFAR100.
    for param in model.parameters():
        # Parameters of newly constructed modules have requires_grad=True by default
        param.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=num_classes)
    std = 0.01
    nn.init.normal_(model.fc.weight, mean=0, std=std)
    nn.init.zeros_(model.fc.bias)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir, augmentation_name=augmentation_name)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    fc_params = model.fc.parameters()
    optimizer = torch.optim.Adam(params=fc_params, lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    loss_module = nn.CrossEntropyLoss()
    loss_module = loss_module.to(device)
    model = model.to(device)

    train_losses = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)

    best_model = None
    best_acc = -np.inf
    for epoch in tqdm(range(epochs)):
        train_losses[epoch] = train_one_epoch(device, model, loss_module, optimizer, train_loader)
        val_acc = evaluate_model(model, val_loader, device)
        if val_acc > best_acc:
            best_model = deepcopy(model)
            best_acc = val_accuracies[epoch]

    # Load the best model on val accuracy and return it.
    model = best_model

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_one_epoch(device, model, loss_module, optimizer, train_loader):
    model.train()
    model = model.to(device)
    train_loss = 0.
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        batch_mean_loss = loss_module(logits, y)
        batch_mean_loss.backward()
        optimizer.step()
        batch_loss = batch_mean_loss * train_loader.batch_size
        train_loss += batch_loss
    return train_loss


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    model = model.to(device)

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    n_correct = 0
    n_all = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            n_all += x.shape[0]
            n_correct += (preds == y).sum().item()

    accuracy = n_correct/n_all
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    pass

    # Set the device to use for training
    pass

    # Load the model
    pass

    # Get the augmentation to use
    pass

    # Train the model
    pass

    # Evaluate the model on the test set
    pass

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'cifar10'],
                        help='Dataset to use.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    set_dataset(kwargs.pop('dataset'))
    main(**kwargs)
