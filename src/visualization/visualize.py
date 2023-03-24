import matplotlib.pyplot as plt
import torch
from astropy.visualization import LogStretch, SqrtStretch, AsinhStretch, SinhStretch

def stretcher(input_value="asinh"):
  if input_value == 'log':
    stretch = LogStretch()
  elif input_value == 'sqrt':
    stretch = SqrtStretch()
  elif input_value == 'asinh':
    stretch = AsinhStretch()
  elif input_value == 'sinh':
    stretch = SinhStretch()
  else:
    raise ValueError('Invalid input value')
  return stretch

def train_plotter(groundtruth, noise, denoised, number=4, str="asinh",size=(10,5)):
    # Create a figure with 3 columns and number rows
    fig, axes = plt.subplots(number, 3, figsize=(size[0], size[1] * number))
    if str:
      stretch = stretcher(str)
    
      if number==1:
            # Plot the ground truth image
            axes[0].imshow(stretch(groundtruth[0].permute(1, 2, 0).detach().cpu().numpy()),cmap="gray")
            axes[0].set_title('Ground Truth')

            # Plot the noisy image
            axes[1].imshow(stretch(noise[0].permute(1, 2, 0).detach().cpu().numpy()), cmap="gray")
            axes[1].set_title('Noisy')

            # Plot the denoised image
            axes[2].imshow(stretch(denoised[0].permute(1, 2, 0).detach().cpu().numpy()), cmap="gray")
            axes[2].set_title('Denoised')    
      else:
      # Loop through the batches and plot each row
        for i in range(number):
            # Plot the ground truth image
            axes[i, 0].imshow(stretch(groundtruth[i].permute(1, 2, 0).detach().cpu().numpy()),cmap="gray")
            axes[i, 0].set_title('Ground Truth')

            # Plot the noisy image
            axes[i, 1].imshow(stretch(noise[i].permute(1, 2, 0).detach().cpu().numpy()), cmap="gray")
            axes[i, 1].set_title('Noisy')

            # Plot the denoised image
            axes[i, 2].imshow(stretch(denoised[i].permute(1, 2, 0).detach().cpu().numpy()), cmap="gray")
            axes[i, 2].set_title('Denoised')
            
      # Set the title of the figure
      for ax in axes.flatten():
          ax.set_xticks([])
          ax.set_yticks([])

      fig.tight_layout()
      fig.suptitle('Ground Truth vs. Noisy vs. Denoised Images')
      
      # Show the plot
      plt.show()
      
    else:
      if number==1:
            # Plot the ground truth image
            axes[0].imshow(groundtruth[0].permute(1, 2, 0).detach().cpu().numpy(),cmap="gray")
            axes[0].set_title('Ground Truth')

            # Plot the noisy image
            axes[1].imshow(noise[0].permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
            axes[1].set_title('Noisy')

            # Plot the denoised image
            axes[2].imshow(denoised[0].permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
            axes[2].set_title('Denoised')    
      else:
      # Loop through the batches and plot each row
        for i in range(number):
            # Plot the ground truth image
            axes[i, 0].imshow(groundtruth[i].permute(1, 2, 0).detach().cpu().numpy(),cmap="gray")
            axes[i, 0].set_title('Ground Truth')

            # Plot the noisy image
            axes[i, 1].imshow(noise[i].permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
            axes[i, 1].set_title('Noisy')

            # Plot the denoised image
            axes[i, 2].imshow(denoised[i].permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
            axes[i, 2].set_title('Denoised')
            
      # Set the title of the figure
      for ax in axes.flatten():
          ax.set_xticks([])
          ax.set_yticks([])

      fig.tight_layout()
      fig.suptitle('Ground Truth vs. Noisy vs. Denoised Images')
      
      # Show the plot
      plt.show()
      