import wandb
import torch
import argparse
from pathing import *
from dotenv import load_dotenv
from utils.dataset import AstroDataset
from utils.compiler import CriterionFactory,OptimizerFactory,MetricFactory,ModelFactory,LRSchedulerFactory
from utils.trainer import Trainer
import warnings
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm
import numpy as np
from visualization.visualize import train_plotter
from matplotlib import pyplot as plt

# Turn off warnings
model_dict= {"name": "BasicUNet", "spatial_dims": 2, "in_channels": 1, "out_channels": 1, "features": [32,32,64,128,256,32], 
          "act": ['LeakyReLU', {'negative_slope': 0.1, 'inplace': True}], "norm": ['instance', {'affine': True}]}
metric_dict=[{"name": "SSIM", "gaussian_kernel": False, "sigma": 1.5, "kernel_size": 5, "reduction": "elementwise_mean", "k1": 0.01, "k2": 0.03},
          {"name": "PSNR", "data_range": None, "base": 10, "reduction": "elementwise_mean"},
          # {name: SNR, zero_mean: False},
          ]
device="cuda"
sliding_window={"roi_size": [512,512], "sw_batch_size": 6, "overlap": 0.25, "sw_device": device, "device": device}

def load_model(model, model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model_state'])
    model.to(device)
    return model

    
def inference(eval_loader, model, metrics):
    model.eval()
    all_metric_dict={}
    mean_metric_dict={}
    for metric in metrics:
        all_metric_dict[f"Preds_{type(metric).__name__}"]=[]
        all_metric_dict[f"Noisy_{type(metric).__name__}"]=[]
    with torch.no_grad():
        for data in tqdm(eval_loader):
            noisy_image, image = data['noisy_image'], data['image']
            noisy_image, image = noisy_image.to(device), image.to(device)
            
            
            inferer = SlidingWindowInferer(**sliding_window
                )
            outputs = inferer(noisy_image,model)
            
            # Measure the SSIM and PSNR metrics
            for metric in metrics:
                name=type(metric).__name__
                all_metric_dict[f"Preds_{name}"].append(metric(outputs, image).item())
                all_metric_dict[f"Noisy_{name}"].append(metric(noisy_image, image).item())
        # Calculate the average metrics
        for metric in metrics:
            name=type(metric).__name__
            temp_list = [x for x in all_metric_dict[f"Preds_{name}"] if x != float('-inf')]
            mean_metric_dict[f"Preds_{name}"]=np.nanmean(temp_list)
            temp_list = [x for x in all_metric_dict[f"Noisy_{name}"] if x != float('-inf')]
            mean_metric_dict[f"Noisy_{name}"]=np.nanmean(temp_list)

        # Print the metrics
    for key, value in mean_metric_dict.items():
        print(f"{key}: {value}")

    torch.cuda.empty_cache()
    
def main():

    
    validation = AstroDataset(query_id=args.query_id, 
                              data_split="validation",
                              batch_size=1,
                              num_workers=4)
    validation_ds = validation.initalizeTorchDataset()
    eval_loader = validation.initalizeTorchLoader(validation_ds)
    
    model_factory = ModelFactory()
    metric_factory= MetricFactory()
    model = model_factory.create_model(model_dict)
    model = load_model(model,args.model_path)
    
    metrics = [metric_factory.create_metric(metric) for metric in metric_dict]
    inference(eval_loader=eval_loader,model=model, metrics=metrics)



if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser(description='Argument Parser')
    # parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    # parser.add_argument('--eval_step', type=int, default=10, help='evaluate every eval_step epochs')
    # parser.add_argument('--lr_scheduler', type=str, default='exponential', help='learning rate scheduler')
    # parser.add_argument('--criterion', type=str, default='L1Loss', help='loss criterion')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    # parser.add_argument('--device', type=str, default='cuda', help='device to train on')
    # parser.add_argument('--model', type=str, default='base_denoise', help='model architecture')
    parser.add_argument('--model_path', type=str, default='/home/dguthruf/Dev/astro-deconv/models/hubble_base/zf50mjj4/model_best.pth', help='evaluation metric')
    parser.add_argument('--query_id', type=str, default='hubble_base', help='query_id / name of data')

    args = parser.parse_args()
    
    try:
        main()
    except Exception as e:
        print(e)
        
        
    