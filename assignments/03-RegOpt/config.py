from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 32
    num_epochs = 10
    initial_learning_rate = 0.0008

    T_mult = 3
    eta_min = 0.0004
    T_0 = 10

    batch_size = 64
    num_epochs = 2
    initial_learning_rate = 0.001
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "T_0": T_0,
        "T_mult": T_mult,
        "eta_min": eta_min,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [ToTensor(), Normalize([0.485, 0.456, 0.4], [0.229, 0.224, 0.2])]
    )
