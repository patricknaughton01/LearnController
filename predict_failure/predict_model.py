import torch
import torch.nn as nn
import random

# multi-layer perceptron
def mlp(input_dim, mlp_dims, last_relu=False, dropout=0):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    # print("mlp_dims", mlp_dims)
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    net = nn.Sequential(*layers)
    return net

class Controller(nn.Module):
    def __init__(self, config, model_type='crossing'):
        super().__init__()
        self.self_state_dim = config.getint(model_type, 'self_state_dim') #6
        # print('config.get(model_type, mlp1_dims)')
        # print(config.get(model_type, 'mlp1_dims').split(', '))
        mlp1_dims = [int(x) for x in config.get(model_type, 'mlp1_dims').split(', ')] #[80]
        # print("mlp1_dims", mlp1_dims)

        input_dim = self.self_state_dim + config.getint(model_type, 'other_state_dim') + config.getint('om', 'cell_num') ** 2 * config.getint('om', 'om_channel_size')
        # other_state_dim = 7;  cell_num = 4(to be squared); om_channel_size = 3; input_dim = 61
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True, dropout=config.getfloat(model_type, 'dropout')) #141 [61, 80]

        mlp2_dims = [int(x) for x in config.get(model_type, 'mlp2_dims').split(', ')] #120
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims, dropout=config.getfloat(model_type, 'dropout')) #200 [80, 120]

        mlp3_dims = [int(x) for x in config.get(model_type, 'mlp3_dims').split(', ')] # 128, 64, 5
        self.mlp3 = mlp(
            mlp2_dims[-1] + self.self_state_dim, mlp3_dims,
            dropout=config.getfloat(model_type, 'dropout')
        ) # [126, 128, 64, 5]

    def forward(self, state):
        """First transform the world coordinates to self-centric
        coordinates and then do forward computation. Computes mus, sigmas,
        and correlation of distribution of final states.

        :param tensor state: tensor of shape (batch_size, # of humans,
            length of a rotated state)
        :return: The predicted mus, sigmas, and correlation as a final state
            given some initial state
            :rtype: tensor

        """
        state = state.reshape((1, *state.shape))
        size = state.shape

        # print("size", size) # [8, 6, 61]
        # print("self.self_state_dim", self.self_state_dim)
        self_state = state[:, 0, :self.self_state_dim]
        # print("self_state.shape", self_state.shape) # [8,6]

        mlp1_output = self.mlp1(state.view((-1, size[2]))) # [?,61]
        mlp2_output = self.mlp2(mlp1_output)
        # concatenate agent's state with global weighted humans' state
        batch_state = self_state.repeat(size[1], 1)     # size[1] is batch size
        joint_state = torch.cat([batch_state, mlp2_output], dim=1)
        pred_t = self.mlp3(joint_state)
        return pred_t
