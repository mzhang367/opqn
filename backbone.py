'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    # basic block 34-layer config. two layers per block with 3*3 kernel size

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)#stride = 2
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)# stride = 1
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), #stride = 2, 64-->128
                nn.BatchNorm2d(self.expansion*planes)   # 1*1 convolution for match dimension
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU(channels)

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)

        return x + short_cut


class resnet20_hashing(nn.Module):
    """
    ONLY ONE FULLY CONNECTED LAYER FOLLOWED BY THE BOTTLENECK
    clf: if classification loss is returned, for ddqh_loss
    """
    def __init__(self, num_layers=64, hashing_bits=48, tanh=False, clf=None, size=7):
        super().__init__()
        assert num_layers in [20, 64], 'spherenet num_layers should be 20 or 64'
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 64:
            layers = [3, 8, 16, 3]
        else:
            raise ValueError('sphere' + str(num_layers) + "is not supported!")
        filter_list = [3, 64, 128, 256, 512]
        if size == 7:
            stride_list = [2, 2, 2, 2]
        else:
            stride_list = [1, 2, 2, 2]
            self.bn0 = nn.BatchNorm2d(filter_list[1])
        block = Block
        self.clf = clf
        self.hashing_bits = hashing_bits
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=stride_list[0])
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=stride_list[1])
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=stride_list[2])
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=stride_list[3])
        self.fc = nn.Linear(512*size*size, 512)
        self.bn = nn.BatchNorm1d(512)
        self.logits = nn.Linear(512, self.hashing_bits)
        self.bn_last = nn.BatchNorm1d(self.hashing_bits)
        self.drop = nn.Dropout()
        self.tanh = tanh
        if self.tanh:
            self.tanh_act = nn.Tanh()

        if self.clf is not None:
            self.classifier = nn.Sequential(
                # nn.Tanh(),
                nn.Linear(self.hashing_bits, self.clf),    # (n, class_num)
                nn.LogSoftmax(dim=1)    # log(softmax(x)) function
            )

    def _make_layer(self, block, inplanes, planes, num_units, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.PReLU(planes))
        for i in range(num_units):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn0(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn(x)
        # x = self.drop(x)
        x = self.logits(x)
        out = self.bn_last(x)
        if self.tanh:
            out = self.tanh_act(out)
        if self.clf:
            clf_x = self.classifier(out)
            return out, clf_x
        return out


class resnet20_pq(nn.Module):

    def __init__(self, num_layers=20, feature_dim=512, channel_max=512, size=7):    # size = 4 for 32*32 size dataset
        super().__init__()
        assert num_layers in [20, 64], 'spherenet num_layers should be 20 or 64'
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 64:
            layers = [3, 8, 16, 3]
        else:
            raise ValueError('sphere' + str(num_layers) + "is not supported!")
        if channel_max == 512:
            filter_list = [3, 64, 128, 256, 512]
            if size == 7:
                stride_list = [2, 2, 2, 2]
            else:
                stride_list = [1, 2, 2, 2]

        else:
            filter_list = [3, 16, 32, 64, 128]
            stride_list = [1, 2, 2, 2]

        block = Block
        self.feature_dim = feature_dim
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=stride_list[0])
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=stride_list[1])
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=stride_list[2])
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=stride_list[3])
        self.bn = nn.BatchNorm1d(channel_max*size*size)
        self.fc = nn.Linear(channel_max*size*size, self.feature_dim)
        self.last_bn = nn.BatchNorm1d(self.feature_dim)
        self.drop = nn.Dropout()

    def _make_layer(self, block, inplanes, planes, num_units, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.PReLU(planes))
        for i in range(num_units):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.drop(x)
        x = self.fc(x)
        out = self.last_bn(x)

        return out


class SphereNet20_pq(nn.Module):
    """
    MODIFIED VERSION OF ABOVE TO HANDLE LARGE INPUT SIZE DATASET
    ONLY ONE FULLY CONNECTED LAYER FOLLOWED BY THE BOTTLENECK
    """
    def __init__(self, num_layers=64, feature_dim=512):
        super().__init__()
        assert num_layers in [20, 64], 'spherenet num_layers should be 20 or 64'
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 64:
            layers = [3, 8, 16, 3]
        else:
            raise ValueError('sphere' + str(num_layers) + "is not supported!")
        filter_list = [3, 64, 128, 256, 512]
        block = Block
        self.feature_dim = feature_dim
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.bn = nn.BatchNorm1d(512*7*7)
        self.fc = nn.Linear(512*7*7, self.feature_dim)
        self.last_bn = nn.BatchNorm1d(self.feature_dim)
        self.drop = nn.Dropout()

    def _make_layer(self, block, inplanes, planes, num_units, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.PReLU(planes))
        for i in range(num_units):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.drop(x)
        x = self.fc(x)
        # x = self.drop(x)
        out = self.last_bn(x)

        return out

class ResNet_q(nn.Module):
    """
    Input size: 32 * 32
    """
    def __init__(self, block, num_blocks, num_seg, split=False, feature_dim=512):
        super(ResNet_q, self).__init__()
        self.feature_dim = feature_dim
        self.in_planes = 64 # always starts from 64 here
        self.split = split
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 64 are num of planes
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # now stride !=1  and 64! 128
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # size of 4*4
        self.fc5 = nn.Linear(512 * 4 * 4, self.feature_dim)
        self.bn2 = nn.BatchNorm1d(self.feature_dim)
        # self.drop = nn.Dropout(p=0.3)
        if self.split:
            self.num_seg = num_seg
            self.len_seg = int(self.feature_dim / self.num_seg)
            self.ListBn = nn.ModuleList([nn.BatchNorm1d(self.len_seg) for i in range(self.num_seg)])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)     # [2, 1, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)   # return self.layer1, self.layer2...

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)# 4*4 the full size of feature map, equal to global average pooling

        """
        comment the below lines if using gloabal avg pooling"""
        out = out.view(out.size(0), -1)
        out = self.fc5(out)

        if self.split:
            out = [self.ListBn[i](out[:, self.len_seg*i: self.len_seg*(i+1)]) for i in range(self.num_seg)]
            out = torch.cat(out, dim=1)
        else:
            out = self.bn2(out)
        return out

def CosQuantNet34(num_seg, split, feature_dim):
    return ResNet_q(BasicBlock, [3, 4, 6, 3], num_seg=num_seg, split=split, feature_dim=feature_dim)


if __name__ == '__main__':

    net = resnet20_pq()
    img = torch.randn(3, 3, 112, 112)
    y = net(img)
    # print(net)
    print(y.size())
