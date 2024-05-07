import math
from DepthWiseConv import DepthWiseConv
from PPB import PPB
from PCSA import PCSA
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath
from ops.bra_legacy import BiLevelRoutingAttention
from _common import Attention, AttentionLePE
from hat_arch import HAT
from ctstrsnformer import CATransformerBlock
from Bsconv import BSConvU
def range_to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def range_to_4d(x):
    b, l, c = x.shape
    h = int(math.sqrt(l))
    w = int(math.sqrt(l))
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        out = self.conv(x)  # B H*W C

        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        out = self.deconv(x) # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        x = range_to_4d(x)

        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops

class BConvolutionalGLU(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.fc1 = nn.Linear(in_features, out_features)
        self.Bsconv = BSConvU(in_features*2,in_features*2,kernel_size=3,padding=1)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features*2, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)
        x = self.act(self.Bsconv(x)) * v
        x = self.drop(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=True,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
        #                          DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
        #                          nn.GELU(),
        #                          nn.Linear(int(mlp_ratio * dim), dim)
        #                          )
        self.mlp = BConvolutionalGLU(in_features=dim,out_features=mlp_ratio*dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x
class biformer_layer(nn.Module):
    def __init__(self,depth,dim):
        super(biformer_layer, self).__init__()
        self.biformer_layer = nn.ModuleList([Block(dim) for i in range(depth)])
    def forward(self,x):
        for blk in self.biformer_layer:
            x = blk(x)
        return x
class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()

        self.relu=nn.ReLU(inplace=True)

        self.tanh=nn.Tanh()
        self.refine2= nn.Conv2d(16, 16, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.refine3= nn.Conv2d(16+3, 3, kernel_size=3,stride=1,padding=1)
        self.upsample = F.upsample_nearest


    def forward(self, x):
        dehaze = self.relu((self.refine2(x)))
        shape_out = dehaze.data.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 128)

        x102 = F.avg_pool2d(dehaze, 64)

        x103 = F.avg_pool2d(dehaze, 32)

        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, dehaze), 1)

        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze
class cts_trans_block(nn.Module):
    def __init__(self,dim,num_head,depeth):
        super(cts_trans_block, self).__init__()
        self.blocks = nn.ModuleList(CATransformerBlock(dim=dim,num_heads=num_head,bias=True) for i in range(depeth))
    def forward(self,x):
        for blk in self.blocks:
            x = blk(x)
        return x
class branch_shallow_enchance(nn.Module):
    def __init__(self):
        super(branch_shallow_enchance, self).__init__()
        self.hat = HAT()
    def forward(self,x):
        out = self.hat(x)
        return out

class Multi_scale_fusion2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Multi_scale_fusion2, self).__init__()
        self.depth_conv_x_w = DepthWiseConv(in_channel*2,in_channel)
        self.depth_conv_y_z = DepthWiseConv(in_channel*2,in_channel)
        self.depth_conv_total_1 = DepthWiseConv(in_channel*2,in_channel)
        self.depth_conv_total_2 = DepthWiseConv(in_channel,3)
        self.trans_cov1 = nn.ConvTranspose2d(in_channel*2, out_channel, kernel_size=2, stride=2)
        self.trans_cov2_1 = nn.ConvTranspose2d(in_channel*4,in_channel*2,kernel_size=2,stride=2)
        self.trans_cov2_2 = nn.ConvTranspose2d(in_channel*2,out_channel,kernel_size=2,stride=2)
        self.trans_cov3_1 = nn.ConvTranspose2d(in_channel*8,in_channel*4,kernel_size=2,stride=2)
        self.trans_cov3_2 = nn.ConvTranspose2d(in_channel*4,in_channel*2,kernel_size=2,stride=2)
        self.trans_cov3_3 = nn.ConvTranspose2d(in_channel*2,out_channel,kernel_size=2,stride=2)
        self.gelu = nn.GELU()
    def forward(self,x,y,z,w):
        y = self.trans_cov1(y)
        y = self.gelu(y)

        z = self.trans_cov2_1(z)
        z = self.gelu(z)
        z = self.trans_cov2_2(z)
        z = self.gelu(z)

        w = self.trans_cov3_1(w)
        w = self.gelu(w)
        w = self.trans_cov3_2(w)
        w = self.gelu(w)
        w = self.trans_cov3_3(w)
        w = self.gelu(w)
        x_w = torch.cat([x,w],1)
        y_z = torch.cat([y,z],1)
        x_w = self.depth_conv_x_w(x_w)
        x_w = self.gelu(x_w)
        y_z = self.depth_conv_y_z(y_z)
        y_z= self.gelu(y_z)
        total = torch.cat([x_w,y_z],1)
        total= self.depth_conv_total_1(total)
        total = self.gelu(total)
        total = self.depth_conv_total_2(total)

        return total

class biformer_layer_unet(nn.Module):
    def __init__(self,embed_dim=32,input_dim=3):
        super(biformer_layer_unet, self).__init__()
        self.input_proj = InputProj(in_channel=3,out_channel=embed_dim)
        self.output_proj = OutputProj(in_channel=embed_dim,out_channel=3)
        self.encoder1 = biformer_layer(depth=2,dim=embed_dim)
        self.encoder1_cts = cts_trans_block(dim=embed_dim,num_head=8,depeth=2)
        self.encoder1_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_encoder1 = PCSA(embed_dim,embed_dim,256)
        self.pooling1 = Downsample(embed_dim,embed_dim*2)
        self.encoder2 = biformer_layer(depth=2,dim=embed_dim*2)
        self.encoder2_cts = cts_trans_block(dim=embed_dim*2, num_head=8, depeth=2)
        self.encoder2_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_encoder2 = PCSA(embed_dim*2,embed_dim*2,128)
        self.pooling2 = Downsample(embed_dim*2,embed_dim*4)

        self.encoder3 = biformer_layer(depth=2,dim=embed_dim*4)
        self.encoder3_cts = cts_trans_block(dim=embed_dim*4, num_head=8, depeth=2)
        self.encoder3_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_encoder3 = PCSA(embed_dim*4,embed_dim*4,64)
        self.pooling3 = Downsample(embed_dim*4,embed_dim*8)
        self.encoder4 = biformer_layer(depth=2,dim=embed_dim*8)
        self.encoder4_cts = cts_trans_block(dim=embed_dim * 8, num_head=8, depeth=2)
        self.encoder4_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_encoder4 = PCSA(embed_dim * 8,embed_dim*8,32)
        self.pooling4 = Downsample(embed_dim*8,embed_dim*16)
        self.bottom = biformer_layer(depth=4,dim=embed_dim*16)
        self.bottom_cts = cts_trans_block(dim=embed_dim*16, num_head=8, depeth=4)
        self.bottom_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_bottom = PCSA(embed_dim * 16,embed_dim*16,16)

        self.upsample1 = Upsample(embed_dim*16,embed_dim*8)
        self.decoder1 = biformer_layer(depth=2,dim=embed_dim*8)
        self.decoder1_cts = cts_trans_block(dim=embed_dim*8, num_head=8, depeth=2)
        self.decoder1_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_decoder1 = PCSA(embed_dim * 8,embed_dim*8,32)
        self.upsample2 = Upsample(embed_dim*8,embed_dim*4)
        self.decoder2 = biformer_layer(depth=2,dim=embed_dim*4)
        self.decoder2_cts = cts_trans_block(dim=embed_dim * 4, num_head=8, depeth=2)
        self.decoder2_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_decoder2 = PCSA(embed_dim*4,embed_dim*4,64)
        self.upsample3 = Upsample(embed_dim*4,embed_dim*2)
        self.decoder3 = biformer_layer(depth=2,dim=embed_dim*2)
        self.decoder3_cts = cts_trans_block(dim=embed_dim * 2, num_head=8, depeth=2)
        self.decoder3_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_decoder3 = PCSA(embed_dim*2,embed_dim*2,128)
        self.upsample4 = Upsample(embed_dim*2,embed_dim)
        self.decoder4 = biformer_layer(depth=2,dim=embed_dim)
        self.decoder4_cts = cts_trans_block(dim=embed_dim , num_head=8, depeth=2)
        self.decoder4_alph = nn.Parameter(torch.ones(1) / 2)
        self.cbam_decoder4 = PCSA(embed_dim,embed_dim,256)
        self.simam = simam_module()
        self.branch_shallow_enchance = branch_shallow_enchance()
        self.multi_fusion_block = Multi_scale_fusion2(32,32)
        self.PPB = PPB(9)
        self.final_conv = nn.Conv2d(9,3,kernel_size=1,bias=False)
    def forward(self,x):


        y = self.branch_shallow_enchance(x)
        x = self.input_proj(x)

        res_encoder1 = x
        cts_encoder1_result = self.encoder1_cts(x)
        x = self.encoder1(x)
        x = self.encoder1_alph * cts_encoder1_result + (1 - self.encoder1_alph) * x
        x = self.cbam_encoder1(x)
        x = x + res_encoder1
        conv0= x
        x = self.pooling1(x)
        res_encoder2 = x
        cts_encoder2_result = self.encoder2_cts(x)
        x = self.encoder2(x)
        x = self.encoder2_alph * cts_encoder2_result + (1-self.encoder2_alph)*x
        x = self.cbam_encoder2(x)
        x = res_encoder2 + x
        conv1 = x

        x = self.pooling2(x)
        res_encoder3 = x
        cts_encoder3_result = self.encoder3_cts(x)
        x = self.encoder3(x)
        x = self.encoder3_alph * cts_encoder3_result + (1 - self.encoder3_alph) * x
        x = self.cbam_encoder3(x)
        x = x + res_encoder3
        conv2 = x

        x = self.pooling3(x)
        res_encoder4 = x
        cts_encoder4_result = self.encoder4_cts(x)
        x = self.encoder4(x)
        x = self.encoder4_alph * cts_encoder4_result + (1 - self.encoder4_alph) *x
        x = self.cbam_encoder4(x)
        x = res_encoder4 + x
        conv3 = x


        x = self.pooling4(x)
        res_bottom = x
        cts_bottom_result = self.bottom_cts(x)
        x = self.bottom(x)
        x = self.bottom_alph*cts_bottom_result + (1 - self.bottom_alph) *x
        x = self.cbam_bottom(x)
        x = x+ res_bottom
        conv0 =  self.simam(conv0)
        conv1 = self.simam(conv1)
        conv2 = self.simam(conv2)
        conv3 = self.simam(conv3)
        muilt_fusion = self.multi_fusion_block(conv0,conv1,conv2,conv3)
        x = self.upsample1(x)
        x = x+conv3
        res_decoder1 = x
        cts_decoder1_result = self.decoder1_cts(x)
        x = self.decoder1(x)
        x = self.decoder1_alph*cts_decoder1_result+(1-self.decoder1_alph)*x
        x = self.cbam_decoder1(x)
        x = x + res_decoder1
        x = self.upsample2(x)
        x = x + conv2
        res_decoder2 = x
        cts_decoder2_result = self.decoder2_cts(x)
        x = self.decoder2(x)
        x = self.encoder2_alph*cts_decoder2_result+(1-self.decoder2_alph)*x
        x = self.cbam_decoder2(x)
        x = x + res_decoder2
        x = self.upsample3(x)
        x = x + conv1
        res_decoder3 = x
        cts_decoder3_result = self.decoder3_cts(x)
        x = self.decoder3(x)
        x = self.decoder3_alph*cts_decoder3_result + (1 - self.decoder3_alph)*x
        x = self.cbam_decoder3(x)
        x = res_decoder3 + x
        x = self.upsample4(x)
        x = x + conv0
        res_decoder4 = x
        cts_decoder4_result = self.decoder4_cts(x)
        x = self.decoder4(x)
        x = self.decoder4_alph * cts_decoder4_result + (1 - self.decoder4_alph)*x
        x = self.cbam_decoder4(x)
        x = x + res_decoder4
        x = self.output_proj(x)
        # out = x + y+muilt_fusion
        out = torch.cat([x,y],dim=1)
        out = torch.cat([out,muilt_fusion],dim=1)
        out = self.PPB(out)
        out = self.final_conv(out)
        return out
