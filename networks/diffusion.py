import os, sys
import torch
import torch.nn as nn

class skip_connection(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        width,
        depth,
        weight_norm=True,
        skip_layer=[],
        relu=False,
        tanh=False
    ):
        super().__init__()

        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):

            if l in self.skip_layer:
                lin = torch.nn.Linear(dims[l] + dims[0], dims[l+1])
            else:
                lin = torch.nn.Linear(dims[l], dims[l+1])

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)
            else:
                torch.nn.init.xavier_uniform_(lin.weight)
                torch.nn.init.zeros_(lin.bias)


            setattr(self, "lin" + str(l), lin)

        if relu:
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.LeakyReLU()

        if tanh:
            self.activation = torch.nn.Tanh()

    def forward(self, input):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, D]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [B, ?]
        """

        batch_size, n_dim = input.shape

        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                #x_mid = x.clone()
                x = torch.cat([x, input], -1)
            #print(l, x.shape)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x