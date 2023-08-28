# import the dependc 📽️
import torch 
from torch import nn
from collections import OrderedDict
from torchsummary import summary

#$ https://arxiv.org/pdf/1505.04597v1.pdf 📄


class Unet(nn.Module) :
    '''
        final_out_channel is the channel of the final output 📺
        features_channel is how many filters (or kernels) we use 🔺

    '''
    def __init__(self, in_channels = 3, final_out_channel = 1 , num_filters_in_theFirst  = 32 ) -> None:
        super().__init__( )

        features = num_filters_in_theFirst # 64 in the paper
        #$ down ⤵️
        self.encoder1 = Unet.conv_block(in_channels , features , 'enc1')
        self.encoder2 = Unet.conv_block(features , features * 2 , 'enc2') 
        self.encoder3 = Unet.conv_block(features * 2 , features  * 4, 'enc3')
        self.encoder4 = Unet.conv_block(features * 4 , features  * 8, 'enc4')
        self.maxpool = nn.MaxPool2d(kernel_size = 2 , stride = 2) #* all the encoders layers have the same maxpool params
        #* this is the bottom block of U-net
        self.bottleneck = Unet.conv_block(features * 8, features * 16, name="bottleneck")

        #$ up ⬆️
        #Transposed convolution https://d2l.ai/chapter_computer-vision/transposed-conv.html
        #* output shape of transposed convolution is Output Shape = (Input Shape - 1) * Stride + Kernel Size - 2 * Padding
        #* output shape of Convlution is Output Shape = [(Input Shape - Kernel Size + 2 * Padding) / Stride] + 1
        self.upconv4 = nn.ConvTranspose2d(features * 16 , features * 8 , kernel_size = 2  , stride = 2)
        self.decoder4 = Unet.conv_block(    
                                            features * 16 , #? features * 16 because we gone concatente look to the model aritcuter
                                            features * 8,
                                            'dec4'                                
                                        )
        self.upconv3 = nn.ConvTranspose2d(features * 8 , features * 4 , kernel_size = 2  , stride = 2)
        self.decoder3 = Unet.conv_block(    
                                            features * 8 , #? features * 16 because we gone concatente look to the model aritcuter
                                            features * 4,
                                            'dec3' 
                                        )
        
        self.upconv2 = nn.ConvTranspose2d(features * 4 , features * 2 , kernel_size = 2  , stride = 2)
        self.decoder2 = Unet.conv_block(    
                                            features * 4 , #? features * 16 because we gone concatente look to the model aritcuter
                                            features * 2,
                                            'dec2' 
                                        )
        
        self.upconv1 = nn.ConvTranspose2d(features * 2 , features  , kernel_size = 2  , stride = 2)
        self.decoder1 = Unet.conv_block(    
                                            features * 2 , #? features * 16 because we gone concatente look to the model aritcuter
                                            features ,
                                            'dec1' 
                                        )
        
        self.conv = nn.Conv2d(in_channels = features , out_channels = 1 , kernel_size = 1)

    
    def forward(self , X ) : # 🚀
        #$ down ⬇️
        encoded1 = self.encoder1(X)
        encoded2 = self.encoder2(self.maxpool(encoded1))
        encoded3 = self.encoder3(self.maxpool(encoded2))
        encoded4 = self.encoder4(self.maxpool(encoded3))
        #$ bottom 🌗
        bottom = self.bottleneck(self.maxpool(encoded4))
        #$ up ⬆️
        decoded4 = self.upconv4(bottom)
        decoded4 = torch.cat((decoded4 , encoded4) , dim = 1)
        decoded4 = self.decoder4(decoded4)
        decoded3 = self.upconv3(decoded4)
        decoded3 = torch.cat((decoded3 , encoded3) , dim = 1)
        decoded3 = self.decoder3(decoded3)
        decoded2 = self.upconv2(decoded3)
        decoded2 = torch.cat((decoded2 , encoded2) , dim = 1)
        decoded2 = self.decoder2(decoded2)
        decoded1 = self.upconv1 ( decoded2 )
        decoded1 = torch.cat ( (decoded1 , encoded1 ) , dim = 1)
        decoded1 = self.decoder1(decoded1)
        final_output = torch.sigmoid(self.conv(decoded1))
       

        return final_output # 🚀


    @staticmethod
    def conv_block(in_channel , features , name) : 
        return nn.Sequential (
            OrderedDict (
                [
                    (
                        name + "conv1" ,
                        nn.Conv2d(in_channels = in_channel , out_channels = features , kernel_size = 3 ,padding = 1 ,bias=False)
                    ),
                    ( name + "norm1" , nn.BatchNorm2d( num_features = features)) , # y=  x−E[x] ∗ γ+β / sqr(Var[x]+ϵ) ⚾
                    (name + 'relu1' , nn.ReLU(inplace = True)),

                    (
                        name + 'conv2',
                        nn.Conv2d(in_channels = features , out_channels = features , kernel_size = 3 ,padding = 1 ,bias=False)
                    ),
                    ( name + "norm2" , nn.BatchNorm2d( num_features = features)) , # y=  x−E[x] ∗ γ+β / sqr(Var[x]+ϵ) ⚾
                    (name + 'relu2' , nn.ReLU(inplace = True)),
                ]
            )
        )



