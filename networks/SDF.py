import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

class SDF(nn.Module):
    def __init__(
        self,
        d_in=3,
        d_out=1,
        dims=[512, 512, 512, 512, 512, 512, 512, 512],
        skip_in=[4],
        geometric_init=True,
        radius_init=1,
        beta=100
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            out_dim = dims[layer + 1]

            if layer != self.num_layers - 2:
                if layer in skip_in:
                    lin = nn.utils.weight_norm(nn.Linear(dims[layer]+d_in, out_dim))
                else:
                    lin = nn.utils.weight_norm(nn.Linear(dims[layer], out_dim))
            else:
                lin = nn.Linear(dims[layer], out_dim)
            

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, input, latent_code):
        # input: (p, 3)
        # latent_code: (p, n_z)

        
        if type(latent_code) != type(None):
            x_in = torch.cat((input, latent_code), dim=-1)
            x = torch.cat((input, latent_code), dim=-1)
        else:
            x_in = input.clone()
            x = input.clone()

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, x_in], -1) / np.sqrt(2)
            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
        
class SDF2branch_deepSDF(nn.Module):
    def __init__(
        self,
        d_in=3,
        d_out=1,
        dims=[512, 512, 512, 512, 512, 512, 512, 512],
        skip_in=[4],
        geometric_init=True,
        radius_init=1,
        beta=100
    ):
        super().__init__()

        self.d_in = d_in
        self.radius_init = radius_init
        dims = [d_in] + dims + [d_out]

        self.lin0 = nn.Linear(dims[0], dims[1])
        self.lin1 = nn.Linear(dims[1], dims[2])
        self.lin2 = nn.Linear(dims[2], dims[3])
        self.lin3_1 = nn.Linear(dims[3]+d_in, dims[4])
        self.lin4_1 = nn.Linear(dims[4], dims[5])
        self.lin5_1 = nn.Linear(dims[5], dims[6])
        self.lin6_1 = nn.Linear(dims[6], dims[6])
        self.lin7_1 = nn.Linear(dims[6], dims[6])
        self.lin8_1 = nn.Linear(dims[6], 1)

        self.lin3_2 = nn.Linear(dims[3]+d_in, dims[4])
        self.lin4_2 = nn.Linear(dims[4], dims[5])
        self.lin5_2 = nn.Linear(dims[5], dims[6])
        self.lin6_2 = nn.Linear(dims[6], d_out-1)

        self.num_layers = len(dims)
        self.skip_in = skip_in

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

        self.initial_weight(dims)

    def initial_weight(self, dims):
        torch.nn.init.normal_(self.lin0.weight, 0.0, np.sqrt(2) / np.sqrt(dims[1]))
        torch.nn.init.constant_(self.lin0.bias, 0.0)
        torch.nn.init.normal_(self.lin1.weight, 0.0, np.sqrt(2) / np.sqrt(dims[2]))
        torch.nn.init.constant_(self.lin1.bias, 0.0)
        torch.nn.init.normal_(self.lin2.weight, 0.0, np.sqrt(2) / np.sqrt(dims[3]))
        torch.nn.init.constant_(self.lin2.bias, 0.0)
        torch.nn.init.normal_(self.lin3_1.weight, 0.0, np.sqrt(2) / np.sqrt(dims[4]))
        torch.nn.init.constant_(self.lin3_1.bias, 0.0)
        torch.nn.init.normal_(self.lin3_2.weight, 0.0, np.sqrt(2) / np.sqrt(dims[4]))
        torch.nn.init.constant_(self.lin3_2.bias, 0.0)
        torch.nn.init.normal_(self.lin4_1.weight, 0.0, np.sqrt(2) / np.sqrt(dims[5]))
        torch.nn.init.constant_(self.lin4_1.bias, 0.0)
        torch.nn.init.normal_(self.lin4_2.weight, 0.0, np.sqrt(2) / np.sqrt(dims[5]))
        torch.nn.init.constant_(self.lin4_2.bias, 0.0)
        torch.nn.init.normal_(self.lin5_1.weight, 0.0, np.sqrt(2) / np.sqrt(dims[6]))
        torch.nn.init.constant_(self.lin5_1.bias, 0.0)
        torch.nn.init.normal_(self.lin5_2.weight, 0.0, np.sqrt(2) / np.sqrt(dims[6]))
        torch.nn.init.constant_(self.lin5_2.bias, 0.0)

        
        torch.nn.init.normal_(self.lin6_1.weight, 0.0, np.sqrt(2) / np.sqrt(dims[6]))
        torch.nn.init.constant_(self.lin6_1.bias, 0.0)
        torch.nn.init.normal_(self.lin7_1.weight, 0.0, np.sqrt(2) / np.sqrt(dims[6]))
        torch.nn.init.constant_(self.lin7_1.bias, 0.0)

        torch.nn.init.constant_(self.lin8_1.bias, -self.radius_init)
        torch.nn.init.normal_(self.lin8_1.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[6]), std=0.00001)
        torch.nn.init.constant_(self.lin6_2.bias, -self.radius_init)
        torch.nn.init.normal_(self.lin6_2.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[6]), std=0.00001)


    def forward(self, input, latent_code):
        # input: (p, 3)
        # latent_code: (p, n_z)

        if type(latent_code) != type(None):
            x_in = torch.cat((input, latent_code), dim=-1)
            x = torch.cat((input, latent_code), dim=-1)
        else:
            x_in = input.clone()
            x = input.clone()

        x = self.activation(self.lin0(x))
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))

        x = torch.cat([x, x_in], -1) / np.sqrt(2)

        x_sdf = self.activation(self.lin3_1(x))
        x_label = self.activation(self.lin3_2(x))
        x_sdf = self.activation(self.lin4_1(x_sdf))
        x_label = self.activation(self.lin4_2(x_label))
        x_sdf = self.activation(self.lin5_1(x_sdf))
        x_label = self.activation(self.lin5_2(x_label))

        
        x_sdf = self.activation(self.lin6_1(x_sdf))
        x_sdf = self.activation(self.lin7_1(x_sdf))

        x_sdf = self.lin8_1(x_sdf)
        x_label = self.lin6_2(x_label)

        x = torch.cat([x_sdf, x_label], -1)

        return x
    
class learnt_representations(nn.Module):
    def __init__(self, rep_size, samples=3):
        super().__init__()
        self._name = 'learnt_representations'
        self.weights = nn.Parameter(torch.zeros(samples, rep_size))

    def forward(self, indexs):
        return self.weights[indexs]
