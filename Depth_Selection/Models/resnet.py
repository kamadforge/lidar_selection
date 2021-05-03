import math
import torch
import torch.nn as nn
from Models.depth_decoder import DepthDecoder
from Models.resnet_encoder import ResnetEncoder


def weights_init(m):
    '''Initialize weights in network
    '''
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def weights_init_bias(m):
    '''Initialize forget gate of lstm
    '''
    if isinstance(m, nn.Conv2d):
        m.bias.data.fill_(1)



def return_tensor(x):
    return x


def define_activation(activation):
    '''Define activation function for state in lstm
    '''
    if activation == 'relu':
        funct = nn.ReLU()
    elif activation == 'elu':
        funct = nn.ELU()
    elif activation == 'tanh':
        funct = nn.Tanh()
    elif activation == 'none':
        funct = return_tensor
    else:
        raise NotImplementedError
    return funct


class ConvLSTMCell(nn.Module):
    ''' Convolutional LSTM module
    '''
    def __init__(self, args, input_channels):
        super(ConvLSTMCell, self).__init__()

        # Define activation function for states in LSTM
        activation_funct = define_activation(args.activation_lstm)
        self.activation_function = activation_funct

        # Define gates
        self.conv_i = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
            )
        if args.activation_lstm != 'none':
            self.conv_g = nn.Sequential(
                nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=True),
                activation_funct
                )
        else:
            self.conv_g = nn.Sequential(
                nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=True),
                )
        self.conv_o = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
            )

        # Apply initialization
        # Default pytorch initialization is kaiming when defining the layer
        # Set biases of forget gate to 1
        self.conv_f.apply(weights_init_bias)

    def forward(self, x, hidden):
        h, c = hidden
        x_lstm = torch.cat((x, h), 1)
        i = self.conv_i(x_lstm)
        f = self.conv_f(x_lstm)
        g = self.conv_g(x_lstm)
        o = self.conv_o(x_lstm)
        c = f * c + i * g
        h = o * self.activation_function(c)
        return (h, c)


class ResNet_EncDec(nn.Module):
    '''Encoder - Decoder network
    '''
    def __init__(self, args):
        super(ResNet_EncDec, self).__init__()

        # Define endcoder and decoder
        self.encoder = ResnetEncoder(args.layers, args.pretrained)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(1), 
                                    num_output_channels=1, use_skips=True)

        # Define init state of LSTM
        input_channels=512
        row, col=6, 20
        h0 = torch.zeros(input_channels, row, col).cuda()
        c0 = torch.zeros(input_channels, row, col).cuda()
        self.h0 = nn.Parameter(h0, requires_grad=True) 
        self.c0 = nn.Parameter(c0, requires_grad=True)

        # Define LSTM
        self.convlstm = ConvLSTMCell(args, input_channels)

        # Define init mode of LSTM
        self.train_init_state = args.train_init_state

    def freeze_bn(self):
        '''Freeze batchnorm layers
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, hidden):
        ''' Inference
        '''
        bs = x.size(0)
        features = self.encoder(x)

        # if self.train_init_state or hidden[0] is None:
            # h = self.h0.repeat(bs, 1, 1, 1)
            # c = self.c0.repeat(bs, 1, 1, 1)
            # hidden = (h, c)

        # hidden = self.convlstm(features[-1], hidden)
        # sliced = features[:-1]
        # sliced.append(hidden[0])
        # output = self.decoder(sliced)

        output = self.decoder(features)
        return output[('depth', 0)], hidden
