import torch

class Dataset(torch.utils.data.Dataset):
  """
    Basic PyTorch Dataset style.
    array_X: numpy array with shape (N cells, F genes)
  """
  def __init__(self, array_X):
    self.X = array_X
    self.length = self.X.shape[0]

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    x = self.X[idx, :]
    return torch.FloatTensor(x)
