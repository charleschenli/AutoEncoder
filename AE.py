import torch

class AE(torch.nn.Module):
  """
    Auto Encoder learns how to represent high dimensional data in low dimension space.

    1. Process: X --> E --> X'
      Given input X with shape (batch size N, higher dimension H),
      self.encoder projects it to E with shape (batch size N, lower dimension L).
      Then, self.decoder projects E to X', X' has the same shape as X.

    2. Mechanism: 
      Apparently, E has less information than X.
      Considering self.decoder does not have access to X,
      if self.decoder can recover X' from E and X' is very similar to X, 
      we can say that E has kept the "important" information from X.

    3. Train: 
      During training, we use optimizer to minimize the distance between X and X'.  

    4. Use:
      When use AE as a dimension reducer, 
      we train an AE with full dataset X, 
      then calculate E with the trained AE.

  """
  def __init__(self, input_size, encoder_size, n_components=2):
    super(). __init__()
    self.activation = torch.nn.ReLU()

    # input_size is usually the number of genes in X, or "how many features that are used to described our sample"
    # n_components has the same meaning to sklearn.decomposition.PCA(n_components), i.e. the number of Principle Components
    self.encoder = torch.nn.Sequential(
        torch.nn.Linear(input_size, encoder_size),
        self.activation,
        torch.nn.Linear(encoder_size, encoder_size),
        self.activation,
        torch.nn.Linear(encoder_size, encoder_size),
        self.activation,
        torch.nn.Linear(encoder_size, n_components),
        self.activation,
    )

    # decoder is basically a reversed encoder
    self.decoder = torch.nn.Sequential(
        torch.nn.Linear(n_components, encoder_size), 
        self.activation,
        torch.nn.Linear(encoder_size, encoder_size),
        self.activation,
        torch.nn.Linear(encoder_size, encoder_size),
        self.activation,
        torch.nn.Linear(encoder_size, input_size),
        self.activation,
    )
    
    # weight tying: 
    #   we teach AE the fact that decoder is a reversed encoder, 
    #   saves it the trouble to find out itself, hopefully this accelerates the convergence
    self.decoder[0].weight = torch.nn.Parameter(self.encoder[6].weight.T)
    self.decoder[2].weight = torch.nn.Parameter(self.encoder[4].weight.T)
    self.decoder[4].weight = torch.nn.Parameter(self.encoder[2].weight.T)
    self.decoder[6].weight = torch.nn.Parameter(self.encoder[0].weight.T)

  def forward(self, x): 
    # (B, H)

    embedded = self.encoder(x)
    # (B, L)

    reconstructed = self.decoder(embedded)
    # (B, H)

    return embedded, reconstructed