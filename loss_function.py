# output shape is (batch_num , 1 , 256 , 256)
# $Dice Loss https://paperswithcode.com/method/dice-loss ⏳
import torch
from torch import nn


#to use forward method
class Jacard_loss  (nn.Module) :
    def __init__(self) -> None:
        super().__init__()

    def forward (self , y , y_hat) :
        assert y.shape == y_hat.shape
        #make the y , y_hat shape like this (batch_size , H , W)
        intersection = ( y * y_hat ).sum()
        union = torch.square(y).sum() + torch.square(y_hat).sum() - intersection

        return 1 - (intersection  / union)
       
