import torch
from torchvision import transforms
from dataset import Dataset_Test
from torch.utils.data import  DataLoader
from U_net import Unet
from utils import plot_output
import argparse

#ò create the dataset
transform = transforms.Compose (
    [
        transforms.ToTensor() ,
        transforms.Resize((256 , 256)),
    ]
)
test_data = Dataset_Test ('test' ,transform )
test_dataLoder = DataLoader (test_data , batch_size = test_data.__len__() , shuffle = True)


#ò test function
def test(num_mask_toplot = None) :
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  with torch.no_grad() :
    model = Unet()
    if device == 'CUDA' :
        model.load_state_dict(torch.load('Wights\modelWights.pth'))
    else : 
       model.load_state_dict(torch.load('Wights\modelWights.pth' , map_location=torch.device('cpu')))
    model.to(device)

  for img in test_dataLoder :
      mask = model(img.to(device))
  if num_mask_toplot :
    return mask [num_mask_toplot] , img [num_mask_toplot]
  else :
    return mask , img

mask , img = test()




def main(args):
    mask, img = test()

    if args.draw_test:
        plot_output(mask.cpu().detach(), img.cpu().detach())
    else:
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run testing and visualization script.")
    parser.add_argument("--draw_test", type=str, help="Specify the output folder for saving visualizations.")

    args = parser.parse_args()
    main(args)




