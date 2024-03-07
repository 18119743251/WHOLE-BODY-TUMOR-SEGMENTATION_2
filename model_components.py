import torch
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from unetr_pp.network_architecture.layers import LayerNorm
from unetr_pp.network_architecture.synapse.transformerblock import TransformerBlock
from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock
from unetr_pp.network_architecture.synapse.dns.MyModules import DenseTransLayer
from unetr_pp.network_architecture.synapse.dns.MyModules import DDPM

einops, _ = optional_import("einops")

class UnetrPPEncoderct(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x,  enc1ct, enc2ct, enc3ct, enc4ct):
        hidden_states = []
        hidden_ct = []
        hidden_ct.append(enc1ct)
        hidden_ct.append(enc2ct)
        hidden_ct.append(enc3ct)
        hidden_ct.append(enc4ct)

        x = self.downsample_layers[0](x)
        x = self.stages[0](x) +  enc1ct

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            if i!=3:
              x = self.stages[i](x) + hidden_ct[i]
            else:
              x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c") + hidden_ct[3]
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x, enc1ct, enc2ct, enc3ct, enc4ct):
        x, hidden_states = self.forward_features(x, enc1ct, enc2ct, enc3ct, enc4ct)
        return x, hidden_states

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        #print(x.shape)
        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states

class CfmthreeConv(nn.Module):
    def __init__(self, channels: int,):
        super().__init__()
        self.conv1 = nn.Conv3d(2*channels, 1*channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(1*channels, 1*channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(3*channels, 1*channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(2*channels, 1*channels, kernel_size=3, stride=1, padding=1)

        self.SG = nn.Sigmoid()

    def forward(self, encpt, encct, fs):
        convcat = torch.cat((encpt, encct), dim=1)
        conv1 = self.conv1(convcat)
        sigconv = self.SG(conv1)

        convf1 = encpt * sigconv
        convf2 = encct * sigconv
        convadd = convf1 + convf2

        conv2 = self.conv2(convadd)
        convcat2 = torch.cat((encpt, conv2, encct), dim=1)
        conv3 = self.conv3(convcat2)

        convcat3 = torch.cat((conv3, fs), dim=1)
        conv4 = self.conv4(convcat3)

        return conv4

class mcmthreeConv(nn.Module):
    def __init__(self, channels: int,):
        super().__init__()
        self.conv1 = nn.Conv3d(2*channels, 1*channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(2*channels, 1*channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(2*channels, 1*channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(2*channels, 1*channels, kernel_size=3, stride=1, padding=1)

        self.SG = nn.Sigmoid()

    def forward(self, encpt, encct, fs):
        convmu1 = encpt * fs * encct
        convmu2 = encpt * fs
        convmu3 = encct * fs

        concat1 = torch.cat((convmu2, fs), dim=1)
        concat2 = torch.cat((convmu3, fs), dim=1)

        convcon1 = self.conv1(concat1)
        convcon2 = self.conv2(concat2)

        concat3 = torch.cat((convcon1, convcon2), dim=1)
        convcon3 = self.conv3(concat3)

        concat4 = torch.cat((convmu1, convcon3), dim=1)
        convcon4 = self.conv4(concat4)

        return convcon4

class UnetrUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()
        #self.convout = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
        
        self.depth_trans = DenseTransLayer(out_channels, out_channels)
        self.selfdc = DDPM(out_channels, out_channels, out_channels, 3, 4)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        #outt = torch.cat([out, skip], dim=1)
        #out = self.convout(outt)
        out = out + skip
        
        #mode = self.depth_trans(skip, encct)
        #out = self.selfdc(out, mode)
        
        out = self.decoder_block[0](out)

        return out
class UnetrUpBlockRGB(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()
        #self.convout = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
        
        self.depth_trans = DenseTransLayer(out_channels, out_channels)
        self.selfdc = DDPM(out_channels, out_channels, out_channels, 3, 4)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip, encct):

        out = self.transp_conv(inp)
        #outt = torch.cat([out, skip], dim=1)
        #out = self.convout(outt)
        out = out + skip
        
        mode = self.depth_trans(skip, encct)
        out = self.selfdc(out, mode)
        
        out = self.decoder_block[0](out)

        return out
        

class UnetrUpBlock_Base_Deta(nn.Module): #这里在输入skip之前已经做过Dase_Detai的处理了，所以不用depth_trans，但是dec的处理不变
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()
        #self.convout = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
        
        self.depth_trans = DenseTransLayer(out_channels, out_channels)
        self.selfdc = DDPM(out_channels, out_channels, out_channels, 3, 4)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip, feature): #inp先升维，然后与skip相加，最后与融合好的feature结合

        out = self.transp_conv(inp)
        #outt = torch.cat([out, skip], dim=1)
        #out = self.convout(outt)
        out = out + skip
        
        out = self.selfdc(out, feature)
        
        out = self.decoder_block[0](out)

        return out
        
class UnetrUpBlockRGB_CT_Atten(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        
        self.decoder_blockct = nn.ModuleList()
        stage_blocks2 = []
        for j in range(depth):
            stage_blocks2.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                 dropout_rate=0.1, pos_embed=True))
        self.decoder_blockct.append(nn.Sequential(*stage_blocks2))
    
        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()
        #self.convout = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
        
        self.depth_trans = DenseTransLayer(out_channels, out_channels)
        self.selfdc = DDPM(out_channels, out_channels, out_channels, 3, 4)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip, encct):

        out = self.transp_conv(inp)
        #outt = torch.cat([out, skip], dim=1)
        #out = self.convout(outt)
        out = out + skip
        
        encct = self.decoder_blockct[0](encct) #do attention for ct to stronger

        mode = self.depth_trans(skip, encct)
        out = self.selfdc(out, mode)
        
        out = self.decoder_block[0](out)

        return out
        
        
class UnetrUpBlockRGB_Block(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()
        #self.convout = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
        
        self.depth_trans = DenseTransLayer(out_channels, out_channels)
        self.selfdc = DDPM(out_channels, out_channels, out_channels, 3, 4)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip, encct):

        out = self.transp_conv(inp)
        #outt = torch.cat([out, skip], dim=1)
        #out = self.convout(outt)
        out = out + skip
        
        mode = self.depth_trans(skip, encct)
        out = self.selfdc(out, mode)
        
        out = self.decoder_block[0](out)

        return out

