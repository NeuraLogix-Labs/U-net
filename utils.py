# 🚔
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import uuid

#$  Un-Normalize Image Tensor opertion 🎁 https://github.com/pytorch/vision/issues/528 🎁
def getInverseNormalize(input , original_mean = (0.485, 0.456, 0.406)  , original_std = (0.229, 0.224, 0.225) ) : 
    if type(original_mean ) and type (original_std) == tuple :
    #$ calculte the new mean and std to return to the orgine value
        inverse_normalize = transforms.Normalize(
                    mean=[-m/s for m, s in zip(original_mean, original_std)],
                    std=[1/s for s in original_std]
                )
    else : 

        inverse_normalize = transforms.Normalize(
                    mean=[-original_mean/original_std],
                    std=[1/original_std]
                )
    
    return inverse_normalize (input)



def plot_output(out_put = None , img = None , output_folder="output") :
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if (type(out_put) == torch.Tensor and type(img) == torch.Tensor )  & (len(out_put.shape) == 3 and len(out_put.shape) == 3):
        # change the type of to numpy array with mask the shpe form of (h , w , channle)
        mask_plt = out_put.permute(1 , 2 , 0).numpy()
        img_plt = img.permute(1 , 2 ,0).numpy() * 255
        plt.imshow(img_plt)
        plt.show()
        plt.imshow(mask_plt)
        plt.show()

    elif (type(out_put) == torch.Tensor and type(img) == torch.Tensor )  & (len(out_put.shape) == 4 and len(out_put.shape) == 4):

      drwing_list = list()

      for i in range(out_put.shape[0]) :
          mask_plt = out_put[i].permute(1 , 2 , 0).numpy()
          img_plt = img[i].permute(1 , 2 ,0).numpy() * 255
          drwing_list.append(img_plt)
          drwing_list.append(mask_plt)

          if len(drwing_list) == 4 :
            fig, grid = plt.subplots(2, 2, figsize=(10, 10))
            plt.tight_layout()

            for ax, im in zip(grid.flat, drwing_list):
              # Iterating over the grid returns the Axes.
              ax.imshow(im)

            # Save the figure
            plt.savefig(f"{output_folder}/{uuid.uuid1()}.png", dpi=300)
            plt.clf()
            del drwing_list
            drwing_list = list()