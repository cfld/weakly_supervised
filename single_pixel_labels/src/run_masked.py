import argparse
import numpy as np
import os
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from datasets import masked_tile_dataloader
from train_masked import train_model
from resnet import resnet50
from UNet2 import ResNetUNet

def set_seeds(seed):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 111)
    _ = torch.cuda.manual_seed(seed + 222)
    _ = random.seed(seed + 333)

def get_pretrained_resnet(model, model_path):
    dim_mlp = model.fc.weight.shape[1]
    model.fc = torch.nn.Sequential(torch.nn.Linear(dim_mlp, dim_mlp), torch.nn.ReLU(), model.fc)
    state_dict = torch.load(model_path)['state_dict']
    for k in list(state_dict.keys()):
        if 'encoder_q' not in k:
            del state_dict[k]

    state_dict = {k.replace('module.', '').replace('encoder_q.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


parser = argparse.ArgumentParser(description="Masked UNet for Remote Sensing Image Segmentation")

# directories
parser.add_argument('--model_dir', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--results_dir', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--n_train', type=int)
parser.add_argument('--label_size', type=int)
parser.add_argument('--pretrained_model_path', default='/home/bjohnson/projects/moco/models/naip/checkpoint_0050.pth.tar')
parser.add_argument('--param_file', default = '/home/ebarnett/weakly_supervised/single_pixel_labels/experiments/params_naip.json')
args = parser.parse_args()

set_seeds(args.seed)

# model and dataset hyperparameters
param_file = os.path.join(args.param_file)

with open(param_file) as f:
    params = json.load(f)

params['label_size'] = args.label_size
params['n_train'] = args.n_train


print("seed =", args.seed, "pretrained =", args.pretrained, "n_train =", args.n_train, "label_size =", args.label_size)

# dataloaders
from glob import glob
files = glob(os.path.join(params['tile_dir'], '*.npy'))

train_val_idx = np.random.choice(np.arange(0,len(files)), params['n_train']+params['n_val'], replace=False)
train_idx = train_val_idx[0:params['n_train']]
val_idx   = train_val_idx[params['n_train']:]

train_files = [os.path.basename(files[i]) for i in train_idx]
val_files   = [os.path.basename(files[i]) for i in val_idx]


dataloaders = {}
dataloaders['train'] = masked_tile_dataloader(params['tile_dir'], 
                                              train_files,
                                              augment=True,
                                              batch_size=params['batch_size'], 
                                              shuffle=params['shuffle'], 
                                              num_workers=params['num_workers'], 
                                              n_samples=params['n_train'],
                                              im_size  = params['label_size'])

dataloaders['val'] = masked_tile_dataloader(params['tile_dir'],
                                            val_files,
                                            augment=False,
                                            batch_size=params['batch_size'],
                                            shuffle=params['shuffle'],
                                            num_workers=params['num_workers'], 
                                            n_samples=params['n_val'],
                                            im_size  = params['label_size'])
dataset_sizes = {}
dataset_sizes['train'] = len(train_files)
dataset_sizes['val'] = len(val_files)

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")


backbone = resnet50(in_channels=params['in_channels'], num_classes=128)
if args.pretrained:
    print("getting pretrained")
    backbone = get_pretrained_resnet(backbone, model_path=args.pretrained_model_path)
backbone = nn.Sequential(*list(backbone.children()))[:-1]
model = ResNetUNet(backbone, n_class=1, in_channels=params['in_channels'])


model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(),
                       lr=params['lr'],
                       betas=(params['beta1'], params['beta2']),
                       weight_decay=params['weight_decay'])

# Decay LR by a factor of gamma every X epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                       step_size=params['decay_steps'],
                                       gamma=params['gamma'])

model, metrics = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, params,
                             num_epochs=params['epochs'],
                             gpu=args.gpu)


results_dir = os.path.join(args.results_dir, f'{args.pretrained}_{params["n_train"]}_{params["label_size"]}_{args.seed}')
os.makedirs(results_dir, exist_ok=True)

if params['save_model']:
    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))

with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)

with open(os.path.join(results_dir, 'params.json'), 'w') as f:
    json.dump(params, f)