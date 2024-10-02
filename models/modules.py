from functools import reduce
import torch
import torch.nn as nn
# from .custom_modules import *

class Gate_module(nn.Module):

    def __init__(self, channels, bottleneck=128):
        super(Gate_module, self).__init__()
        d = max(channels // 2, 32)  #
        self.out_channels = bottleneck
        # self.nb_input = nb_input
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(nn.Conv1d(bottleneck, d, 1, bias=False),
                                 nn.BatchNorm1d(d),
                                 nn.ReLU(inplace=True))  #
        self.fc2 = nn.Conv1d(d, bottleneck * 3, 1, 1, bias=False)  #
        self.softmax=nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size(0)
        s = self.aap(input)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 3, self.out_channels)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(3, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1), a_b))

        return a_b

class Bottleneck(nn.Module):
    """
    Bottleneck block of ResNeXt architectures[1].
    Dynamic scaling policy (DSP) is based on the elastic module[2].

    Reference:
    [1] Xie, Saining, et al.
    "Aggregated residual transformations for deep neural networks." CVPR. 2017.
    [2] Wang, Huiyu, et al.
    "Elastic: Improving cnns with dynamic scaling policies." CVPR. 2019.
    """

    cardinality = 32

    def __init__(self, inplanes, planes, dsp, up_path, gate, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        self.dsp = dsp
        self.up_path = up_path
        self.gate = gate
        cardinality = Bottleneck.cardinality  # 32
        bottel_plane = planes

        if self.dsp:
            cardinality = cardinality // 2  # 16
            bottel_plane = bottel_plane // 2  # 128  256
            cardinality_split = cardinality  # 16
            bottel_plane_split = bottel_plane  # 128  256

            if self.up_path:
                cardinality_split = cardinality_split // 2  # 8
                bottel_plane_split = bottel_plane_split // 2  # 64  128

        # if change in number of filters

        if inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )

        self.conv1 = nn.Conv1d(inplanes, bottel_plane,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottel_plane)
        self.conv2 = nn.Conv1d(bottel_plane, bottel_plane, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm1d(bottel_plane)
        self.conv3 = nn.Conv1d(bottel_plane, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        if self.dsp:
            self.conv1_d = nn.Conv1d(inplanes, bottel_plane_split, kernel_size=1, bias=False)

            self.bn1_d = nn.BatchNorm1d(bottel_plane_split)

            self.conv2_d = nn.Conv1d(
                bottel_plane_split, bottel_plane_split, kernel_size=3, stride=1,
                padding=2,dilation=dilation+1, bias=False, groups=cardinality_split)
            self.bn2_d = nn.BatchNorm1d(bottel_plane_split)
            self.conv3_d = nn.Conv1d(
                bottel_plane_split, planes, kernel_size=1, bias=False)


            if self.up_path:
                self.conv1_u = nn.Conv1d(
                    inplanes, bottel_plane_split, kernel_size=1, bias=False)
                self.bn1_u = nn.BatchNorm1d(bottel_plane_split)

                self.conv2_u = nn.Conv1d(
                    bottel_plane_split, bottel_plane_split, kernel_size=3, stride=1, padding=3, dilation=dilation+2, bias=False,
                    groups=cardinality_split
                )
                self.bn2_u = nn.BatchNorm1d(bottel_plane_split)

                self.conv3_u = nn.Conv1d(
                    bottel_plane_split, planes, kernel_size=1, bias=False)


                if self.gate:self.gate_moduel = Gate_module(planes, planes )
            else:

                if self.gate: self.gate_moduel = Gate_module(planes, planes // 2, nb_input=2)


    def forward(self, x, residual=None):
        if residual is None:
            residual = self.shortcut(x) if hasattr(self, "shortcut") else x

        out = self.conv1(x)
        out = self.conv2(self.relu(self.bn1(out)))
        out = self.conv3(self.relu(self.bn2(out)))

        if self.dsp:

            out_d = self.conv1_d(x)
            out_d = self.conv2_d(self.relu(self.bn1_d(out_d)))
            out_d = self.conv3_d(self.relu(self.bn2_d(out_d)))

            if self.up_path:
                out_u = self.conv1_u(x)
                out_u = self.conv2_u(self.relu(self.bn1_u(out_u)))
                out_u = self.conv3_u(self.relu(self.bn2_u(out_u)))

                # agregation of features using gate module
                if self.gate:
                    # out_cat = torch.cat((out, out_d, out_u), 1)
                    out_sum = [out, out_d, out_u]
                    U = reduce(lambda x, y: x + y, out_sum)
                    a_b_c = self.gate_moduel(U)
                    V = list(map(lambda x, y: x * y, out_sum, a_b_c))
                    out = reduce(lambda x, y: x + y, V)
                # agregation of features using element-wise summation
                else:
                    out += out_d + out_u

            else:
                if self.gate:
                    out_cat = torch.cat((out, out_d), 1)
                    out = self.gate_moduel(out_cat)
                else:
                    out += out_d

        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out
