from box import Box
import torch
import torch.nn as nn

def load_model(config: Box, input_dim: int, n_outputs: int) -> nn.Module:
    if config.model_type == "mlp":
        return SimpleMLP(
            input_dim,
            n_outputs,
            config.mlp_hidden_dim,
            config.mlp_nb_hidden
        )





class SimpleMLP(nn.Module):
   """Basic Multi-Layer Perceptron: Linear layers + ReLU.

   Args:
       input_dim: Flattened input size (2 * T)
       n_outputs: Number of target outputs (n_hb + 3)
       hidden_dim: Size of every hidden layer
       nb_hidden: Number of hidden layers (excluding input and output)
   """

   def __init__(
       self,
       input_dim: int,
       n_outputs: int,
       hidden_dim: int = 256,
       nb_hidden: int = 2,
   ):
       super().__init__()

       layers = []
       in_dim = input_dim

       # Build hidden layers with constant width
       for _ in range(nb_hidden):
           layers.append(nn.Linear(in_dim, hidden_dim))
           layers.append(nn.ReLU())
           in_dim = hidden_dim

       # Final output layer
       layers.append(nn.Linear(in_dim, n_outputs))

       self.net = nn.Sequential(*layers)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       # x: [N, 2, T] -> flatten to [N, 2*T]
       x = x.flatten(start_dim=1)
       return self.net(x)