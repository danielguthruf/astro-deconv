import torch
import monai.losses as losses
import torch.optim as optim
import torchmetrics
import monai.networks.nets as nets
import torchgeometry as tgm

class CriterionFactory:
    def __init__(self):
        self.losses = {
            'L1Loss': torch.nn.L1Loss,
            'MSELoss': torch.nn.MSELoss,
            'SSIMLoss': tgm.losses.SSIM,
            'HuberLoss': torch.nn.HuberLoss,
        }
    
    def create_criterion(self, loss_dict):
        loss_name = loss_dict.pop('name')
        if loss_name not in self.losses:
            raise ValueError(f"Unknown loss function name: {loss_name}")
        return self.losses[loss_name](**loss_dict)
    
class OptimizerFactory:
    def __init__(self, model_params):
        self.optimizers = {
            'Adam': optim.Adam,
            'NAdam': optim.NAdam,
            'AdamW': optim.AdamW,
            'Adagrad': optim.Adagrad,
            'Adamax': optim.Adamax,
        }
        self.model_params=model_params
    
    def create_optimizer(self, optimizer_dict):
        optimizer_name = optimizer_dict.pop('name')
        if optimizer_name not in self.optimizers:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")
        if optimizer_name == 'WeightedAdam':
            class_weights = optimizer_dict.pop('class_weights', None)
            return self.optimizers[optimizer_name](**optimizer_dict, weight=torch.tensor(class_weights))
        else:
            return self.optimizers[optimizer_name](self.model_params,**optimizer_dict)
        


class MetricFactory:
    def __init__(self,device="cuda"):
        self.metrics = {
            'Accuracy': torchmetrics.Accuracy,
            'Precision': torchmetrics.Precision,
            'Recall': torchmetrics.Recall,
            'F1': torchmetrics.F1Score,
            'AUC': torchmetrics.AUROC,
            'MSE': torchmetrics.MeanSquaredError,
            'MAE': torchmetrics.MeanAbsoluteError,
            'SSIM': torchmetrics.StructuralSimilarityIndexMeasure,
            'PSNR': torchmetrics.PeakSignalNoiseRatio,
            'SNR': torchmetrics.SignalNoiseRatio,
        }
        self.device=device
        
    def create_metric(self, metric_dict):
        metric_name = metric_dict.pop('name')
        if metric_name not in self.metrics:
            raise ValueError(f"Unknown metric function name: {metric_name}")
        return self.metrics[metric_name](**metric_dict).to(self.device)
    
    

class ModelFactory:
    def __init__(self):
        self.models = {
            'BasicUNet': nets.BasicUNet,
            'BasicUNetPlusPlus': nets.BasicUNetPlusPlus,
            'FlexibleUNet': nets.FlexibleUNet,
        }
    
    def create_model(self, model_dict):
        model_name = model_dict.pop('name')
        if model_name not in self.models:
            raise ValueError(f"Unknown model function name: {model_name}")
        return self.models[model_name](**model_dict)
    
    

class LRSchedulerFactory:
    def __init__(self,optimizer):
        self.lr_schedulers = {
            'ExponentialLR': optim.lr_scheduler.ExponentialLR,
        }
        self.optimizer=optimizer
    
    def create_lr_scheduler(self, lr_scheduler_dict):
        lr_scheduler_name = lr_scheduler_dict.pop('name')
        if lr_scheduler_name not in self.lr_schedulers:
            raise ValueError(f"Unknown lr scheduler function name: {lr_scheduler_name}")
        return self.lr_schedulers[lr_scheduler_name](self.optimizer,**lr_scheduler_dict)