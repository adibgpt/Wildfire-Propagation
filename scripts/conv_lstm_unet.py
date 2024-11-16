# ConvLSTM-UNet model architecture definition

import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvLSTM Cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, device):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = self.kernel_size // 2
        self.device = device

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        o = torch.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(self.device),
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(self.device),
        )

# ConvLSTM Module
class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, device):
        super(ConvLSTM, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                ConvLSTMCell(input_size, cur_input_dim, hidden_dim, kernel_size, bias=True, device=device)
            )

    def forward(self, x, hidden_state):
        b, t, c, h, w = x.size()
        seq_len = t

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            outputs = []

            for time_step in range(seq_len):
                h, c = self.layers[layer_idx](x[:, time_step, :, :, :], (h, c))
                outputs.append(h)

            x = torch.stack(outputs, dim=1)

        return x, hidden_state

    def init_hidden(self, batch_size):
        hidden_states = []
        for layer in self.layers:
            hidden_states.append(layer.init_hidden(batch_size))
        return hidden_states

# Double Convolution Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# UNet Architecture
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = self._down(64, 128)
        self.down2 = self._down(128, 256)
        self.down3 = self._down(256, 512)
        self.down4 = self._down(512, 1024)
        self.up1 = self._up(1024, 512)
        self.up2 = self._up(512, 256)
        self.up3 = self._up(256, 128)
        self.up4 = self._up(128, 64)
        self.outc = nn.Conv2d(64, output_channels, kernel_size=1)

    def _down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def _up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(torch.cat([x4, self.up1(x5)], dim=1))
        x = self.up2(torch.cat([x3, self.up2(x)], dim=1))
        x = self.up3(torch.cat([x2, self.up3(x)], dim=1))
        x = self.up4(torch.cat([x1, self.up4(x)], dim=1))
        return self.outc(x)

# ConvLSTM-UNet Model
class ConvLSTM_UNet(nn.Module):
    def __init__(self, input_size, input_channels, output_channels, hidden_dim, kernel_size, num_layers, device):
        super(ConvLSTM_UNet, self).__init__()
        self.convlstm = ConvLSTM(input_size, input_channels, hidden_dim, kernel_size, num_layers, device)
        self.unet = UNet(input_channels=hidden_dim, output_channels=output_channels)

    def forward(self, x, hidden_state):
        convlstm_out, _ = self.convlstm(x, hidden_state)
        return self.unet(convlstm_out[:, -1, :, :, :])

    def init_hidden(self, batch_size):
        return self.convlstm.init_hidden(batch_size)
