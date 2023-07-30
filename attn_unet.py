import torch
import torch.nn as nn


class UTnet_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UTnet_enc(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = UTnet_conv(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))  # downsample

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)


class UTnet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024), p_dropout=0.0):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][
            :-1
        ]  # number of channels in the decoder # [::-1] reverse, [:-1] select all until last
        self.enc_blocks = nn.ModuleList(
            [
                UTnet_conv(in_ch, out_ch)
                for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])
            ]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(
            2
        )  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # deconvolution
        self.dec_blocks = nn.ModuleList(
            [
                UTnet_conv(in_ch, out_ch)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()
        )  # 1x1 convolution for producing the output
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            x = self.dropout(x)
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(
            self.dec_blocks, self.upconvs, enc_features[::-1]
        ):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


class UTNet_attn(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wx = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0), nn.BatchNorm2d(out_c)
        )
        self.Wskip = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0), nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0), nn.Sigmoid()
        )

    def forward(self, x, skip):
        Wx = self.Wx(x)
        Wskip = self.Wskip(skip)
        out = self.relu(Wx + Wskip)
        out = self.conv(out)
        return out * skip


class UTNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024), p_dropout=0.0):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][
            :-1
        ]  # number of channels in the decoder # [::-1] reverse, [:-1] select all until last
        self.enc_blocks = nn.ModuleList(
            [
                UTnet_conv(in_ch, out_ch)
                for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])
            ]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(
            2
        )  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
                for in_ch, _ in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # deconvolution
        self.attns = nn.ModuleList(
            [
                UTNet_attn(in_ch, out_ch)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                UTnet_conv(in_ch + out_ch, out_ch)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()
        )  # 1x1 convolution for producing the output
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            x = self.dropout(x)
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution

        x = self.enc_blocks[-1](x)

        # decode
        for block, upconv, feature, attn in zip(
            self.dec_blocks, self.upconvs, enc_features[::-1], self.attns
        ):
            x = upconv(x)  # increase resolution
            feature = attn(x, feature)  # attn
            x = torch.cat([x, feature], dim=1)  # concatenate skip features\
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


if __name__ == "__main__":
    x = torch.randn((8, 3, 400, 400))
    model = UTNet()
    output = model(x)
    print(output.shape)
