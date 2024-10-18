from omegaconf import OmegaConf
from models.common.mixed_attn_block import CAB
from models.common.mixed_attn_block_efficient import MyAttention, _get_stripe_info
import parameter
from models.common.ops import get_relative_coords_table_all, get_relative_position_index_simple, calculate_mask, \
    calculate_mask_all, blc_to_bchw, bchw_to_blc
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

parameter._init()


# 残差单元
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # Resudual connect: fn(x) + x
        return self.fn(x, **kwargs) + x


# 层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # using Layer Normalization before input to fn layer
        return self.fn(self.norm(x), **kwargs)


# 前馈网络
class FeedForward(nn.Module):
    # Feed Forward Neural Network
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Two linear network with GELU and Dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(H * W))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)
        # print(x.shape,'aaa')
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # print(x.shape,'bbb')
        weight = torch.view_as_complex(self.complex_weight)
        # print(x.shape, weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        # print(x.shape)
        x = x.reshape(B, C, H, W)

        return x


class Block(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.norm = norm_layer(dim)

    def forward(self, x):
        return self.norm(self.filter(x))


class MyCDFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, hw):
        super().__init__()
        self.filter = out_channels
        self.se = GlobalFilter(dim=dim, h=hw, w=hw // 2 + 1)
        self.conv2D = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, hsi, sar):
        sar = self.conv2D(sar)
        jc = hsi * sar
        jc = self.se(jc)
        jd = torch.abs(hsi - sar)
        ja = jc * hsi + jc * sar
        jf = ja + jd
        return hsi + jf, sar + jf


class Mixformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            mlp_dim,
            dropout,
            input_resolution,
            num_heads_window,
            num_heads_stripe,
            window_size,
            stripe_size,
            stripe_groups,
            stripe_shift,
            qkv_bias=True,
            qkv_proj_type="linear",
            anchor_proj_type="avgpool",
            anchor_one_stage=True,
            anchor_window_down_factor=1,
            attn_drop=0.0,
            pretrained_window_size=[0, 0],
            pretrained_stripe_size=[0, 0],
            out_proj_type="linear",
            local_connection=False,
            euclidean_dist=False
    ):
        super().__init__()

        args = OmegaConf.create(
            {
                "out_proj_type": out_proj_type,
                "local_connection": local_connection,
                "euclidean_dist": euclidean_dist,
            }
        )
        self.local_connection = local_connection
        self.norm = nn.LayerNorm(dim)
        if self.local_connection:
            self.conv = CAB(dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            # using multi-self-attention and feed forward neural network repeatly
            attn = MyAttention(
                dim=dim,
                input_resolution=input_resolution,
                num_heads_w=num_heads_window,
                num_heads_s=num_heads_stripe,
                window_size=window_size,
                window_shift=i % 2 == 0,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] if stripe_shift else False,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                attn_drop=attn_drop,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                args=args
            )
            self.layers.append(nn.ModuleList([
                # Residual(PreNorm(dim, attn)),
                attn,
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, x_size, table_index_mask):
        for attn, ff in self.layers:
            x = attn(x, x_size, table_index_mask)
            x = x + self.norm(x)
            if self.local_connection:
                x = x + self.conv(x, x_size)
            x = ff(x)
        return x


class MyBlock(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            dim,
            depth,
            mlp_dim,
            channels,
            num_heads_window,
            num_heads_stripe,
            window_size=2,
            stripe_size=[2, 2],
            stripe_groups=[None, None],
            stripe_shift=False,
            qkv_bias=True,
            qkv_proj_type="linear",
            anchor_proj_type="avgpool",
            anchor_one_stage=True,
            anchor_window_down_factor=1,
            attn_drop=0.0,
            pretrained_window_size=[0, 0],
            pretrained_stripe_size=[0, 0],
            local_connection=False,
            dropout=0.,
            emb_dropout=0.,
    ):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, dim))

        self.input_resolution = to_2tuple(image_size)
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_window_size = pretrained_window_size
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor

        self.to_embedding = nn.Linear(self.patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        for k, v in self.set_table_index_mask(self.input_resolution).items():
            self.register_buffer(k, v)

        self.transformer = Mixformer(
            dim,
            depth,
            mlp_dim,
            dropout,
            image_size,
            num_heads_window,
            num_heads_stripe,
            window_size,
            stripe_size,
            stripe_groups,
            stripe_shift,
            qkv_bias,
            qkv_proj_type,
            anchor_proj_type,
            anchor_one_stage,
            anchor_window_down_factor,
            attn_drop,
            pretrained_window_size,
            pretrained_stripe_size,
            local_connection=local_connection
        )
        self.embedding_to = nn.Linear(dim, self.patch_dim)

    def forward(self, x):
        p = self.patch_size
        b, c, h, w = x.shape

        x_size = (x.shape[2], x.shape[3])
        table_index_mask = self.get_table_index_mask(x.device, x_size)

        hh = int(h / p)
        embed = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # print(self.patch_dim, self.dim, embed.size())
        embed = self.to_embedding(embed)
        b, n, c = embed.shape
        embed += self.pos[:, :n]
        embed = self.dropout(embed)
        # print('embed', embed.size())
        embed = self.transformer(embed, x_size, table_index_mask)
        x = self.embedding_to(embed)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=hh, p1=p, p2=p)
        return x

    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        # ss - stripe_size, sss - stripe_shift_size
        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df = self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(
            ss[::-1], self.pretrained_stripe_size, df
        )

        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == self.input_resolution:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask
        
class MyNet(nn.Module):
    def __init__(self, out_features, window_size=[4, 3, 2], depths=[1, 1, 1]):
        super(MyNet, self).__init__()
        self.out_features = out_features

        self.window_size1 = to_2tuple(window_size[0])
        self.window_size2 = to_2tuple(window_size[1])
        self.window_size3 = to_2tuple(window_size[2])

        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        num_heads = 6

        self.WaveBlock1 = MyBlock(image_size=8,
                                  patch_size=1,

                                  dim=180,
                                  depth=depths[0],
                                  mlp_dim=360,

                                  channels=64,
                                  num_heads_window=num_heads,
                                  num_heads_stripe=num_heads,

                                  window_size=self.window_size1,
                                  stripe_size=[4, 4],
                                  stripe_shift=False,
                                  anchor_window_down_factor=2,
                                  local_connection=True,
                                  dropout=0.,
                                  emb_dropout=0
                                  )

        self.WaveBlock2 = MyBlock(image_size=6,
                                  patch_size=1,
                                  dim=180,
                                  depth=depths[1],
                                  mlp_dim=360,

                                  channels=128,
                                  num_heads_window=num_heads,
                                  num_heads_stripe=num_heads,

                                  window_size=self.window_size2,
                                  stripe_size=[3, 3],
                                  stripe_shift=False,
                                  anchor_window_down_factor=1,
                                  local_connection=False,
                                  dropout=0.,
                                  emb_dropout=0
                                  )

        self.WaveBlock3 = MyBlock(image_size=4,
                                  patch_size=1,
                                  dim=180,
                                  depth=depths[2],
                                  mlp_dim=360,

                                  channels=256,
                                  num_heads_window=num_heads,
                                  num_heads_stripe=num_heads,

                                  window_size=self.window_size3,
                                  stripe_size=[2, 2],
                                  stripe_shift=False,
                                  anchor_window_down_factor=1,
                                  local_connection=False,
                                  dropout=0.,
                                  emb_dropout=0
                                  )

        self.CDFBlock1 = MyCDFBlock(4, 64, 64, 8)
        self.CDFBlock2 = MyCDFBlock(64, 128, 128, 6)
        self.CDFBlock3 = MyCDFBlock(128, 256, 256, 4)

        self.drop_hsi = nn.Dropout(0.6)
        self.drop_sar = nn.Dropout(0.6)
        self.drop_fusion = nn.Dropout(0.6)

        self.fusionlinear_hsi = nn.Linear(in_features=1024, out_features=self.out_features)
        self.fusionlinear_sar = nn.Linear(in_features=1024, out_features=self.out_features)
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, hsi, sar):
        hsi = hsi.reshape(-1, hsi.shape[1] * hsi.shape[2], hsi.shape[3], hsi.shape[4])

        hsi_feat1 = self.hsi_conv1(hsi)
        hsi_feat1 = hsi_feat1 + self.WaveBlock1(hsi_feat1)

        hsi_feat1, sar_feat1 = self.CDFBlock1(hsi_feat1, sar)

        hsi_feat2 = self.hsi_conv2(hsi_feat1)
        hsi_feat2 = hsi_feat2 + self.WaveBlock2(hsi_feat2)

        hsi_feat2, sar_feat2 = self.CDFBlock2(hsi_feat2, sar_feat1)

        hsi_feat3 = self.hsi_conv3(hsi_feat2)

        hsi_feat3 = hsi_feat3 + self.WaveBlock3(hsi_feat3)

        hsi_feat3, sar_feat3 = self.CDFBlock3(hsi_feat3, sar_feat2)
        # print('3,',hsi_feat3.size())
        hsi_feat4 = hsi_feat3.reshape(-1, hsi_feat3.shape[1], hsi_feat3.shape[2] * hsi_feat3.shape[3])
        sar_feat4 = sar_feat3.reshape(-1, sar_feat3.shape[1], sar_feat3.shape[2] * sar_feat3.shape[3])

        # fusion_feat = torch.cat((hsi_feat4, sar_feat4), dim=1)
        # print('4,',hsi_feat4.size())
        hsi_feat = F.max_pool1d(hsi_feat4, kernel_size=4)
        hsi_feat = hsi_feat.reshape(-1, hsi_feat.shape[1] * hsi_feat.shape[2])
        # print('5,',hsi_feat.size())
        sar_feat = F.max_pool1d(sar_feat4, kernel_size=4)
        sar_feat = sar_feat.reshape(-1, sar_feat.shape[1] * sar_feat.shape[2])
        # fusion_feat = F.max_pool1d(fusion_feat, kernel_size=4)
        # fusion_feat = fusion_feat.reshape(-1, fusion_feat.shape[1] * fusion_feat.shape[2])

        hsi_feat = self.drop_hsi(hsi_feat)
        sar_feat = self.drop_sar(sar_feat)
        # fusion_feat = self.drop_fusion(fusion_feat)
        # print('6,',hsi_feat.size())
        output_hsi = self.fusionlinear_hsi(hsi_feat)
        output_sar = self.fusionlinear_sar(sar_feat)

        weights = torch.sigmoid(self.weight)
        outputs = weights[0] * output_hsi + weights[1] * output_sar
        return outputs
