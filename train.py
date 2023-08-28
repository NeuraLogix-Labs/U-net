# ⭕
import argparse
import torch
from torch.optim import Adam
import tqdm
from U_net import Unet
from loss_function import Jacard_loss
from dataset import Dataset_Train
from torch.utils.data import  DataLoader
from torchvision import transforms



transform = transforms.Compose (
    [
        transforms.ToTensor() ,
        transforms.Resize((256 , 256)),
    ]
)

label_transform = transforms.Compose (
    [
        transforms.ToTensor() ,
        transforms.Resize((256 , 256)),
    ]
)


train_data = Dataset_Train ( transform ,label_transform =  label_transform )
train_dataLoder = DataLoader (train_data , batch_size = 64 , shuffle = True)


def train(num_epoch , lr  = 0.01) :
    #ò to be tracking the loss
    model = Unet()

    #ò check this link for survey of loss functions for semantic segmentation
         #ò https://arxiv.org/pdf/2006.14822.pdf

    loss_fn = Jacard_loss()
    
    optimizer = Adam(model.parameters() , lr = lr)
    loop_bar = tqdm(train_dataLoder)
    #? got the device cpu or gpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range (num_epoch) :
        model.train()
        for img , mask in loop_bar :
            #$ change the input and mask device if cuda is avilble
            img.to(device) , mask.to(device) # 💻

            optimizer.zero_grad()
            mask_pred = model(img)
            loss = loss_fn (mask ,mask_pred )
            loss.backward()
            optimizer.step()
            loop_bar.set_postfix(loss=loss.item())
        
        model.eval()
        for img in test_dataLoder : 
            mask_pred = model(img)
            loss = loss_fn (mask ,mask_pred )
            list_loss.append(loss)



# 🚋🚋🚋🚋

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Training Script")
    parser.add_argument("--num_epoch", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="modelWeights.pth", help="Name of the saved model file")
    args = parser.parse_args()

    train(args.num_epoch, args.lr, args.model_name)

if __name__ == "__main__":
    main()