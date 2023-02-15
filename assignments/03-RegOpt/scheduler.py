from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import math


class CustomLRScheduler(_LRScheduler):
    """
    The customized Learning Rate Scheduler:

    Arguments:
        _LRScheduler: lr scheduler

    """

    def __init__(self, optimizer, T_0, T_mult, eta_min, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        Arguments:
            T_0: number of iterations for the restart
            T_mult: A factor increases
            eta_min: Minimum learning rate
        """
        # ... Your Code Here ...

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.cycle = 0
        self.cycle_length = T_0
        self.last_restart = 0
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Return:
            the learning rate for each epoch
        """

        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]

        if self.last_epoch == self.cycle_length:
            self.last_restart = self.last_epoch
            self.cycle += 1
            self.cycle_length = int(self.T_0 * (self.T_mult**self.cycle))

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.last_restart) / self.cycle_length
                )
            )
            / 2
            for base_lr in self.base_lrs
        ]
