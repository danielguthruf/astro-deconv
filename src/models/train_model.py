import wandb
import argparse
from pathing import *
from dotenv import load_dotenv
from utils.dataset import AstroDataset
from utils.compiler import CriterionFactory,OptimizerFactory,MetricFactory,ModelFactory,LRSchedulerFactory
from utils.trainer import Trainer
import warnings

# Turn off warnings
warnings.filterwarnings("ignore")

def merge_dicts():
    pass
def set_wandb(wandb_run_name=None, wandb_id=None):
    if wandb_id:
        run=wandb.init(
            id=wandb_id,
            resume="allow",
            )
    else:
        wandb_run_name = f"tmp_{args.query_id}"
        wandb_id = wandb.util.generate_id()
        run=wandb.init(
            config=f"{CONFIG_PATH}/{args.config}.yaml",
            name=wandb_run_name,
            id=wandb_id,
            )

    return run

def end_wandb():
    wandb.finish()
    
    
def main():
    train = AstroDataset(query_id=args.query_id, 
                         data_split="train",
                         batch_size=run.config.train_batchsize,
                         num_workers=run.config.num_workers,
                         crop=run.config.crop)
    train_ds = train.initalizeTorchDataset()
    train_loader = train.initalizeTorchLoader(train_ds)
    
    validation = AstroDataset(query_id=args.query_id, 
                              data_split="validation",
                              batch_size=run.config.validation_batchsize,
                              num_workers=run.config.num_workers)
    validation_ds = validation.initalizeTorchDataset()
    eval_loader = validation.initalizeTorchLoader(validation_ds)
    
    model_factory = ModelFactory()
    criterion_factory = CriterionFactory()
    metric_factory= MetricFactory()
    model = model_factory.create_model(run.config.model)
    
    optimizer_factory = OptimizerFactory(model.parameters())
    
    metrics = [metric_factory.create_metric(metric) for metric in run.config.metric]
    criterion = criterion_factory.create_criterion(run.config.loss)
    optimizer = optimizer_factory.create_optimizer(run.config.optimizer)
    lr_scheduler_factory = LRSchedulerFactory(optimizer=optimizer)
    lr_scheduler = lr_scheduler_factory.create_lr_scheduler(run.config.lr_scheduler)
    run.watch(model, log='all')
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=run.config.device,
        lr_scheduler=lr_scheduler,
        model_dir=f"{MODEL_PATH}/{args.query_id}/{run.id}",
        evaluation_step=run.config.evaluation_step,
        metricS = metrics,
        run=run,
    )
    trainer.train_loop(run.config.epochs,load_best_model=args.finetune,load_last_model=run.resumed)


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
    # parser.add_argument('--metric', type=str, default='hello', help='evaluation metric')
    parser.add_argument('--query_id', type=str, default='hubble_base', help='query_id / name of data')
    parser.add_argument('--wandb_id', type=str, default="c5wa3d3g", help="run_id of wandb_run")
    parser.add_argument('--run_name', type=str, default=None, help="specific_run_name")
    parser.add_argument('--finetune', type=str, default=None, help="specific_run_name")
    parser.add_argument('--config', type=str, default="default", help="name of you config file yaml")

    args = parser.parse_args()
    run=set_wandb(args.run_name,args.wandb_id)
    
    try:
        main()
        end_wandb()
    except Exception as e:
        print(e)
        end_wandb()
        
        
    