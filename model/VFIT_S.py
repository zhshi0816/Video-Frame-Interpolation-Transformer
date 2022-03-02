import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Sep_STS_Encoder import ResBlock

def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1

class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)

class upSplit(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3,3), stride=(1,2,2), padding=1),
                 ]
            )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x, output_size):
        x = self.upconv[0](x, output_size=output_size)
        return x

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)


class UNet_3D_3D(nn.Module):
    def __init__(self, n_inputs=4, joinType="concat", ks=5, dilation=1):
        super().__init__()

        nf = [192, 128, 64, 32]
        ws = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        nh = [2, 4, 8, 16]
        self.joinType = joinType
        self.n_inputs = n_inputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)

        from model.Sep_STS_Encoder import SepSTSEncoder
        self.encoder = SepSTSEncoder(nf, n_inputs, window_size=ws, nh=nh)

        self.decoder = nn.Sequential(
            upSplit(nf[0], nf[1]),
            upSplit(nf[1]*growth, nf[2]),
            upSplit(nf[2]*growth, nf[3]),
        )

        def SmoothNet(inc, ouc):
            return torch.nn.Sequential(
                Conv_3d(inc, ouc, kernel_size=3, stride=1, padding=1, batchnorm=False),
                ResBlock(ouc, kernel_size=3),
            )

        nf_out = 64
        self.smooth_ll = SmoothNet(nf[1]*growth, nf_out)
        self.smooth_l = SmoothNet(nf[2]*growth, nf_out)
        self.smooth = SmoothNet(nf[3]*growth, nf_out)

        self.predict_ll = SynBlock(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=True)
        self.predict_l = SynBlock(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=False)
        self.predict = SynBlock(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=False)

    def forward(self, frames):
        images = torch.stack(frames, dim=2)
        _, _, _, H, W = images.shape

        ## Batch mean normalization works slightly better than global mean normalization, thanks to https://github.com/myungsub/CAIN
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_

        x_0, x_1, x_2, x_3, x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4, x_3.size()))
        dx_3 = joinTensors(dx_3 , x_3 , type=self.joinType)

        dx_2 = self.lrelu(self.decoder[1](dx_3, x_2.size()))
        dx_2 = joinTensors(dx_2 , x_2 , type=self.joinType)

        dx_1 = self.lrelu(self.decoder[2](dx_2, x_1.size()))
        dx_1 = joinTensors(dx_1 , x_1 , type=self.joinType)

        fea3 = self.smooth_ll(dx_3)
        fea2 = self.smooth_l(dx_2)
        fea1 = self.smooth(dx_1)

        out_ll = self.predict_ll(fea3, frames, x_2.size()[-2:])

        out_l = self.predict_l(fea2, frames, x_1.size()[-2:])
        out_l = F.interpolate(out_ll, size=out_l.size()[-2:], mode='bilinear') + out_l

        out = self.predict(fea1, frames, x_0.size()[-2:])
        out = F.interpolate(out_l, size=out.size()[-2:], mode='bilinear') + out

        if self.training:
            return out_ll, out_l, out
        else:
            return out

class MySequential(nn.Sequential):
    def forward(self, input, output_size):
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):
                input = module(input, output_size)
            else:
                input = module(input)
        return input


class SynBlock(nn.Module):
    def __init__(self, n_inputs, nf, ks, dilation, norm_weight=True):
        super(SynBlock, self).__init__()

        def Subnet_offset(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                nn.Softmax(1) if norm_weight else nn.Identity()
            )

        def Subnet_occlusion():
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=nf, out_channels=n_inputs, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        self.n_inputs = n_inputs
        self.kernel_size = ks
        self.kernel_pad = int(((ks - 1) * dilation) / 2.0)
        self.dilation = dilation

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        import cupy_module.adacof as adacof
        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

        self.ModuleWeight = Subnet_weight(ks ** 2)
        self.ModuleAlpha = Subnet_offset(ks ** 2)
        self.ModuleBeta = Subnet_offset(ks ** 2)
        self.moduleOcclusion = Subnet_occlusion()

        self.feature_fuse = Conv_2d(nf * n_inputs, nf, kernel_size=1, stride=1, batchnorm=False, bias=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, fea, frames, output_size):
        H, W = output_size

        occ = torch.cat(torch.unbind(fea, 1), 1)
        occ = self.lrelu(self.feature_fuse(occ))
        Occlusion = self.moduleOcclusion(occ, (H, W))

        B, C, T, cur_H, cur_W = fea.shape
        fea = fea.transpose(1, 2).reshape(B*T, C, cur_H, cur_W)
        weights = self.ModuleWeight(fea, (H, W)).view(B, T, -1, H, W)
        alphas = self.ModuleAlpha(fea, (H, W)).view(B, T, -1, H, W)
        betas = self.ModuleBeta(fea, (H, W)).view(B, T, -1, H, W)

        warp = []
        for i in range(self.n_inputs):
            weight = weights[:, i].contiguous()
            alpha = alphas[:, i].contiguous()
            beta = betas[:, i].contiguous()
            occ = Occlusion[:, i:i+1]
            frame = F.interpolate(frames[i], size=weight.size()[-2:], mode='bilinear')

            warp.append(
                occ * self.moduleAdaCoF(self.modulePad(frame), weight, alpha, beta, self.dilation)
            )

        framet = sum(warp)
        return framet

if __name__ == '__main__':
    model = UNet_3D_3D('unet_18', n_inputs=4, n_outputs=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))
    # inp = [torch.randn(1, 3, 225, 225).cuda() for i in range(4)]
    # out = model(inp)
    # print(out[0].shape, out[1].shape, out[2].shape)
