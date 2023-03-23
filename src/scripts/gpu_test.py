import torch
from pathing import *
# setting device on GPU if available, else CPU
num_gpus=torch.cuda.device_count()
print(f"Num-GPUs: {num_gpus}")
for i in range(num_gpus):
    device = torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()


    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(i))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
