import os
import sys
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torchvision

dir_name = '.'
fig_dir = os.path.join(dir_name, 'figs')
model_dir = os.path.join(dir_name, 'models')

def load_data(train_batch_size, test_batch_size):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4
    )
    datasets = (train_dataset, test_dataset)
    loaders = (train_loader, test_loader)
    
    return datasets, loaders

def plot_examples(dataset, shape=(10,10), output=None):
    x, y = shape
    plt.figure(figsize=(8,8))
    for i in range(1, x*y+1): 
        plt.subplot(x, y, i)
        plt.imshow(dataset.data[i-1], cmap='gray')
        plt.axis("off")    
    if output:
        plt.savefig(os.path.join(fig_dir, 'ejemplos.pdf'), dpi=300)
    plt.show()

def create_base_model_name(model_type, optimizer_name, hidden_layer_size, 
                           p_dropout, train_batch_size, 
                           learning_rate, momentum=None):
    
    base_model_name = '{}_{}_hidden{:04}_dropout{:.3f}_batch{:05}_lr{:.6f}'.format(
        model_type, optimizer_name, hidden_layer_size, p_dropout, 
        train_batch_size, learning_rate
    )
    if optimizer_name == 'SGD' and momentum:
        base_model_name += '_momentum{:.6f}'.format(momentum)
        
    return base_model_name