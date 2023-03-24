from statistics import mean
import torch
import wandb
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
from visualization.visualize import train_plotter
from matplotlib import pyplot as plt
from monai.inferers import SlidingWindowInferer
class Trainer:
    def __init__(self, 
                 model, 
                 criterion, 
                 optimizer, 
                 train_loader, 
                 eval_loader,
                 metricS, 
                 device='cuda',
                 lr_scheduler=None, 
                 model_dir='saved_models',
                 evaluation_step=10,
                 run=None,
                 half=False,
                 ):
        
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.scaler = GradScaler()
        self.lr_scheduler=lr_scheduler
        self.model_dir=model_dir
        self.evaluation_step=evaluation_step
        self.run = run
        self.metricS=metricS
        self.half = half

        os.makedirs(model_dir, exist_ok=True)

        self.best_eval_metric = float('-inf')
        # self.best_model_path = best_model_path
        # self.last_model_path = last_model_path

    def save_model(self, epoch, pre="last"):
        model_path = os.path.join(self.model_dir, f'model_{pre}.pth')
        if not os.path.exists(model_path.split(f"/model_{pre}")[0]):
            os.makedirs(model_path.split(f"/model_{pre}")[0])
        state = {
            'epoch': epoch,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
        }
        torch.save(state, model_path)

        return model_path

    def save_best_model(self, eval_metric, epoch):
        if eval_metric > self.best_eval_metric:
            self.best_eval_metric = eval_metric
            self.best_model_path = self.save_model(epoch, "best")

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler_state'])
        self.optimizer.param_groups[0]['lr']=state_dict['learning_rate']
        self.model.to(self.device)
        
        
        print(f"Loaded model from {model_path}")
        
        
        return state_dict['epoch']
    
    def train(self, epoch):
        
        self.model.train()

        running_loss = 0.0
        for i, data in (enumerate(self.train_loader, 0)):
            start_time = time.time()
            noisy_image, image = data['noisy_image'], data['image']
            noisy_image, image = noisy_image.to(self.device), image.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            loss = self.backpropagation(noisy_image=noisy_image,image=image)
            running_loss += loss.item()

            # Print loss and time for each step
            current_time = time.time()
            time_elapsed = current_time - start_time
            formatted_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}, Time: {time_elapsed:.2f} seconds, Current time: {formatted_time}")

            
        self.run.log({'epoch': epoch, 'train_loss': running_loss / i, 'learning_rate': self.optimizer.param_groups[0]['lr']})
        running_loss = 0.0

        torch.cuda.empty_cache()


    def backpropagation(self,noisy_image,image):
        if self.half:
            with autocast():
                outputs = self.model(noisy_image)
                loss = self.criterion(outputs, image)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs=self.model(noisy_image)
            loss = self.criterion(outputs, image)
            loss.backward()
            self.optimizer.step()
        return loss
    def evaluate(self, epoch):
        self.model.eval()
        all_metric_dict={}
        mean_metric_dict={}
        for metric in self.metricS:
            all_metric_dict[f"Preds_{type(metric).__name__}"]=[]
            if self.evaluation_step==epoch:
                all_metric_dict[f"Noisy_{type(metric).__name__}"]=[]
        with torch.no_grad():
            for data in tqdm(self.eval_loader):
                noisy_image, image = data['noisy_image'], data['image']
                noisy_image, image = noisy_image.to(self.device), image.to(self.device)
                if self.half:
                    with autocast():
                        inferer = SlidingWindowInferer(**self.run.config.sliding_window
                        )
                        outputs = inferer(noisy_image,self.model)
                else:
                        inferer = SlidingWindowInferer(**self.run.config.sliding_window
                        )
                        outputs = inferer(noisy_image,self.model)       
                # Measure the SSIM and PSNR metrics
                for metric in self.metricS:
                    name=type(metric).__name__
                    all_metric_dict[f"Preds_{name}"].append(metric(outputs, image).item())
                    if self.evaluation_step==epoch:
                        all_metric_dict[f"Noisy_{name}"].append(metric(noisy_image, image).item())
            # Calculate the average metrics
            for metric in self.metricS:
                name=type(metric).__name__
                temp_list = [x for x in all_metric_dict[f"Preds_{name}"] if x != float('-inf')]
                mean_metric_dict[f"Preds_{name}"]=np.nanmean(temp_list)
                if self.evaluation_step==epoch:
                    temp_list = [x for x in all_metric_dict[f"Noisy_{name}"] if x != float('-inf')]
                    mean_metric_dict[f"Noisy_{name}"]=np.nanmean(temp_list)

            # Print the metrics
        for key, value in mean_metric_dict.items():
            print(f"{key}: {value}")
        self.run.log(mean_metric_dict)
        self.save_best_model(mean_metric_dict["Preds_PeakSignalNoiseRatio"], epoch)

        torch.cuda.empty_cache()

    def train_loop(self, num_epochs, start_epoch=1, load_last_model=None, load_best_model=False):
        if load_best_model:
            start_epoch =self.load_model(f"{self.model_dir}/model_best.pth")
        if load_last_model:
            start_epoch =self.load_model(f"{self.model_dir}/model_last.pth")
            
    
        
        torch.backends.cudnn.benchmark = True
        for epoch in range(start_epoch, num_epochs + 1):
            start_epoch = time.time()

            self.train(epoch)
            time_elapsed=time.time()-start_epoch
            print(f"Epoch {epoch}, Train-Time: {time_elapsed:.2f}" )
            if epoch % self.evaluation_step == 0:
                start_eval=time.time()
                self.evaluate(epoch)
                time_elapsed=time.time()-start_eval
                print(f"Epoch {epoch}, Eval-Time: {time_elapsed:.2f}" )
                # if self.lr_scheduler is not None:
                #     self.lr_scheduler.step(epoch)
            _=self.save_model(epoch)

