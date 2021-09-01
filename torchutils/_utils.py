from torch.utils.data import Subset
import numpy as np


def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]

