import torch
import torch.nn as nn
import numpy as np

class CNN_in_paer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = [nn.Conv2d(3, 64, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(True)]
        self.conv0 = nn.Sequential(*self.conv0)

        self.conv1 = [nn.Conv2d(64, 128, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(True)]
        self.conv1 = nn.Sequential(*self.conv1)

        self.conv2 = [nn.Conv2d(128, 256, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(True)]
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = [nn.Conv2d(256, 512, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(True)]
        self.conv3 = nn.Sequential(*self.conv3)

        self.conv4 = [nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(True)]
        self.conv4 = nn.Sequential(*self.conv4)

        self.fc1 = [nn.Flatten(), nn.Linear(4096, 512), nn.ReLU(True)]
        self.fc1 = nn.Sequential(*self.fc1)

        self.fc2 = [nn.Linear(512, 50), nn.Tanh()]
        self.fc2 = nn.Sequential(*self.fc2)

        self.output = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output(x)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.num_gpus = len(self.gpulist)

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]

        if self.num_gpus == 1:
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        elif self.num_gpus == 2:
            model1 = []
            mult = 2**n_downsampling
            mid = int(n_blocks / 2)
            for i in range(mid):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(n_blocks - mid):
                model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        elif self.num_gpus == 3:
            model1 = []
            model2 = []
            mult = 2**n_downsampling
            mid1 = int(n_blocks / 5)
            mid2 = mid1 + int((n_blocks - mid1) / 4.0 * 3)
            # mid = int(n_blocks / 2)
            for i in range(mid1):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(mid1, mid2):
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(mid2, n_blocks):
                model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        if self.num_gpus >= 2:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        self.act]
            model1 += [nn.ReflectionPad2d(3)]
            model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model1 += [nn.Tanh()]
        else:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model0 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        self.act]
            model0 += [nn.ReflectionPad2d(3)]
            model0 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model0 += [nn.Tanh()]

        self.model0 = nn.Sequential(*model0)
        # self.model0.cuda(self.gpulist[0])
        if self.num_gpus == 2:
            self.model1 = nn.Sequential(*model1)
            self.model1.cuda(self.gpulist[1])
        if self.num_gpus == 3:
            self.model2 = nn.Sequential(*model2)
            self.model2.cuda(self.gpulist[2])

    def forward(self, input):
        input = input
        input = self.model0(input)
        if self.num_gpus == 3:
            input = input.cuda(self.gpulist[2])
            input = self.model2(input)
        if self.num_gpus == 2:
            input = input.cuda(self.gpulist[1])
            input = self.model1(input)
        return input


class CNN(nn.Module):
    def __init__(self, input_nc, output_nc, nf=64):
        super(CNN, self).__init__()
        net = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, nf, kernel_size=7),
               nn.BatchNorm2d(nf), nn.ReLU(True)]
        model = []
        model.append(net)

        self.n_downsample = 2

        mult = 1
        for i in range(self.n_downsample):
            mult = 2**i
            net = [nn.Conv2d(nf*mult, nf*mult*2, kernel_size=3, stride=2, padding=1),
                     nn.BatchNorm2d(nf*mult*2), nn.ReLU(True)]
            model.append(net)

        net = [nn.Conv2d(nf*mult*2, 1, kernel_size=1), nn.BatchNorm2d(1), nn.ReLU(True)]
        net += [nn.AdaptiveAvgPool2d((7,7)), nn.Flatten()]
        net += [nn.Linear(49, output_nc)]

        model.append(net)

        for i in range(2+self.n_downsample):
            setattr(self,'model'+str(i), nn.Sequential(*model[i]))

if __name__ == '__main__':
    x = np.random.random((3, 64, 64))
    x = torch.from_numpy(x).unsqueeze(0).type(torch.float32)
    print(x.size())
    model = CNN_in_paer()
    print(model(x))