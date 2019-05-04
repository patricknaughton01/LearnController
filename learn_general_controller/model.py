import torch
import torch.nn as nn

# multi-layer perceptron
def mlp(input_dim, mlp_dims, last_relu=False, dropout=0):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    print("mlp_dims", mlp_dims)
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
        self.global_state_dim = mlp1_dims[-1] #80
        # print("global_state_dim", self.global_state_dim)

        input_dim = self.self_state_dim + config.getint(model_type, 'other_state_dim') + config.getint('om', 'cell_num') ** 2 * config.getint('om', 'om_channel_size')
        # other_state_dim = 7;  cell_num = 4(to be squared); om_channel_size = 3; input_dim = 61 
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True, dropout=config.getfloat(model_type, 'dropout')) #141 [61, 80]

        mlp2_dims = [int(x) for x in config.get(model_type, 'mlp2_dims').split(', ')] #120
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims, dropout=config.getfloat(model_type, 'dropout')) #200 [80, 120]
        self.with_global_state = config.get(model_type, 'with_global_state') #true

        attention_dims = [int(x) for x in config.get(model_type, 'attention_dims').split(', ')] #64, 32, 1
        if self.with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims) # [160, 64, 32, 1]
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        rnn_input_dim = mlp2_dims[-1] + self.self_state_dim # 86
        
        rnn_hidden_dim = config.getint(model_type, 'rnn_hidden_dim') #256
        self.rnn_cell = nn.LSTMCell(input_size=rnn_input_dim, hidden_size=rnn_hidden_dim)

        mlp3_dims = [int(x) for x in config.get(model_type, 'mlp3_dims').split(', ')] # 128, 64, 5
        self.mlp3 = mlp(rnn_hidden_dim, mlp3_dims, dropout=config.getfloat(model_type, 'dropout')) # [256, 128, 64, 5]
        self.attention_weights = None

    def forward(self, state, h_t=None):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim] # what does self state store ??
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        
        # print("size", size)
        # print("self_state", self_state.shape)
        # print("mlp1 inp", state.view((-1, size[2])).shape )
        # print("mlp1 output", mlp1_output.shape)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        # print('calling forward')

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        h_t = self.rnn_cell(joint_state, h_t)
        pred_t = self.mlp3(h_t[0])
        # print('pred_t',pred_t.shape)
        # print('h_t')
        # print(h_t)
        return pred_t, h_t
    