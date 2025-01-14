import math
import torch
import torch.nn as nn
import torchvision
from . import block as B
from . import spectral_norm as SN

####################
# Generator
####################


class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=5, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        # Upsampling: 2x 업스케일링 두 번, 1.25x 업스케일링 한 번 수행
        if upsample_mode == 'upconv':
            upsampler = [
                B.upconv_blcok(nf, nf, upscale_factor=2, act_type=act_type),
                B.upconv_blcok(nf, nf, upscale_factor=2, act_type=act_type),
                B.upconv_blcok(nf, nf, upscale_factor=5/4, act_type=act_type)
            ]
        elif upsample_mode == 'pixelshuffle':
            upsampler = [
                B.pixelshuffle_block(nf, nf, upscale_factor=2, act_type=act_type),
                B.pixelshuffle_block(nf, nf, upscale_factor=2, act_type=act_type),
                B.pixelshuffle_block(nf, nf, upscale_factor=5/4, act_type=act_type)
            ]
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        # Model definition
        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x



#class RRDBNet(nn.Module):
#    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=5, norm_type=None, \
#            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
#        super(RRDBNet, self).__init__()
#        
#        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
#        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
#            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
#        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
#
#        if upsample_mode == 'upconv':
#            # 2x 업스케일링 두 번, 1.25x 업스케일링 한 번 수행
#            upsampler = [
#                B.upconv_blcok(nf, nf, upscale_factor=2, act_type=act_type),
#                B.upconv_blcok(nf, nf, upscale_factor=2, act_type=act_type),
#                B.upconv_blcok(nf, nf, upscale_factor=5/4, act_type=act_type)
#            ]
#        elif upsample_mode == 'pixelshuffle':
#            upsampler = [
#                B.pixelshuffle_block(nf, nf, upscale_factor=2, act_type=act_type),
#                B.pixelshuffle_block(nf, nf, upscale_factor=2, act_type=act_type),
#                B.pixelshuffle_block(nf, nf, upscale_factor=5/4, act_type=act_type)
#            ]
#        else:
#            raise NotImplementedError('Upsample mode [{:s}] is not supported'.format(upsample_mode))
#        
#        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
#        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
#
#        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
#            *upsampler, HR_conv0, HR_conv1)
#
#    def forward(self, x):
#        x = self.model(x)
#        return x

# DenseBlock 변경 실험
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=5, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        
        # Initial Convolution Block
        self.conv_first = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        # Replacing RRDB with DenseBlock
        current_channels = nf
        dense_blocks_list = []
        
        for _ in range(nb):
            dense_blocks_list.append(DenseBlock(current_channels, gc, num_layers=4))
            current_channels += 4 * gc  # Update channel count after each dense block
        
        self.dense_blocks = nn.ModuleList(dense_blocks_list)

        # LR Conv layer after Dense Blocks
        self.LR_conv = B.conv_block(current_channels, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        # Upsampling layers
        if upsample_mode == 'upconv':
            # 2x upscaling twice, then 1.25x upscaling
            self.upsampler = nn.Sequential(
                B.upconv_blcok(nf, nf, upscale_factor=2, act_type=act_type),
                B.upconv_blcok(nf, nf, upscale_factor=2, act_type=act_type),
                B.upconv_blcok(nf, nf, upscale_factor=5/4, act_type=act_type)
            )
        elif upsample_mode == 'pixelshuffle':
            self.upsampler = nn.Sequential(
                B.pixelshuffle_block(nf, nf, upscale_factor=2, act_type=act_type),
                B.pixelshuffle_block(nf, nf, upscale_factor=2, act_type=act_type),
                B.pixelshuffle_block(nf, nf, upscale_factor=5/4, act_type=act_type)
            )
        else:
            raise NotImplementedError('Upsample mode [{:s}] is not supported'.format(upsample_mode))
        
        # Final Convolution layers for HR Image
        self.HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x):
        # Initial feature extraction
        fea = self.conv_first(x)
        
        # Pass through all dense blocks
        for dense_block in self.dense_blocks:
            fea = dense_block(fea)

        # Apply LR conv layer
        fea = self.LR_conv(fea)

        # Upsampling and reconstruction
        out = self.upsampler(fea)
        out = self.HR_conv0(out)
        out = self.HR_conv1(out)

        return out  # Return the high-resolution image output
    
import torch
import torch.nn as nn
from . import block as B

# Reconstruction Module Definition
class ReconstructionModule(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(ReconstructionModule, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upscale_factor, stride=upscale_factor)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.ca = ChannelAttention(out_channels)  # Channel Attention Mechanism
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # 1x1 Conv for high-res info preservation
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.ca(x) * x  # Apply Channel Attention
        x = self.conv1x1(x)
        return self.norm(x)

# Channel Attention Class Definition
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        out = avg_out + max_out
        return self.sigmoid(out)

# Channel Instance Residual Block (CIRB) Definition
class CIRB(nn.Module):
    def __init__(self, channels):
        super(CIRB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.ca = ChannelAttention(channels)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.ca(out) * out
        return self.relu(out + identity)

# Dense Layer Definition
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(growth_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], 1)

# Dense Block Definition
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = [DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class WS_SRNet_Generator(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, growth_rate=32, gc=32, upscale=5,
                 norm_type='instance', act_type='leakyrelu', mode='CNA'):
        
        super(WS_SRNet_Generator, self).__init__()

        # Initial Convolution Block
        self.conv_first = B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)

        # Dense Blocks for Feature Extraction
        current_channels = nf
        dense_blocks_list = []
        
        for _ in range(nb):
            dense_blocks_list.append(DenseBlock(current_channels, gc, num_layers=4))
            current_channels += 4 * gc
        
        self.dense_blocks = nn.ModuleList(dense_blocks_list)

        # Feature Refinement with CIRBs
        refinement_layers = [CIRB(current_channels) for _ in range(3)]
        
        self.refinement = nn.Sequential(*refinement_layers)

        # Upsampling with Reconstruction Module and CIRB
        upsampler_layers = [
            ReconstructionModule(current_channels, nf, upscale_factor=upscale),
            #CIRB(nf)
        ]
        
        self.upsampler = nn.Sequential(*upsampler_layers)

        # Final Convolution Layers to get HR Image
        self.conv_hr = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type) 
        self.conv_last = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x):
        # Initial feature extraction
        fea = self.conv_first(x)
        # Pass through all dense blocks
        for dense_block in self.dense_blocks:
            fea = dense_block(fea)
        # Feature refinement through CIRBs
        out = self.refinement(fea)
        # Upsampling and reconstruction
        out = self.upsampler(out)
        # Final convolution layers
        out = self.conv_hr(out)
        out = self.conv_last(out)
        return out  # Return the high-resolution image output


####################
# Discriminator
####################


# VGG style Discriminator with input size 100x100
class Discriminator_VGG_128(nn.Module):  # 100x100으로 수정함
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)

        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100),  # 512 * 3 * 3 = 4608로 수정
            nn.LeakyReLU(0.2, True), 
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output