"""Implements a two layer Neural Network."""

from torch.nn import Module, Linear, ReLU
from src.utils.mapper import configmapper


@configmapper.map("models", "two_layer_nn")
class TwoLayerNN(Module):
    """Implements two layer neural network.

    Methods:
        forward(x_input): Returns the output of the neural network.
    """

    def __init__(self, embedding, dims):
        """Construct the two layer Neural Network.

        This method is used to initialize the two layer neural network,
        with a given embedding type and corresponding arguments.

        Args:
            embedding (torch.nn.Module): The embedding layer for the model.
            dims (list): List of dimensions for the neural network, input to output.
        """
        super(TwoLayerNN, self).__init__()

        self.embedding = embedding
        self.linear1 = Linear(dims[0], dims[1])
        self.relu = ReLU()
        self.linear2 = Linear(dims[1], dims[2])

    def forward(self, x_input):
        """
        Return the output of the neural network for an input.

        Args:
            x_input (torch.Tensor): The input tensor to the neural network.

        Returns:
            x_output (torch.Tensor): The output tensor for the neural network.
        """
        output = self.embedding(x_input)
        output = self.linear1(output)
        output = self.relu(output)
        x_output = self.linear2(output)
        return x_output
