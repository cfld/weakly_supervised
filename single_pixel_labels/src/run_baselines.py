import argparse
import numpy as np
import os
import random
import json

import torch

from datasets import masked_tile_dataloader

from sklearn.ensemble import RandomForestClassifier as RF

def set_seeds(seed):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 111)
    _ = torch.cuda.manual_seed(seed + 222)
    _ = random.seed(seed + 333)


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

metrics = {}

inputs, labels, masks = [], [], []
for i, (im, lab, ma) in enumerate(dataloaders['train']):
    if i == 0:
        inputs = im
        labels = lab
        masks = ma
    else:
        inputs = torch.cat((inputs, im), 0)
        labels = torch.cat((labels, lab), 0)
        masks = torch.cat((masks, ma), 0)


locs = np.where(masks == True)

model = RF(n_estimators=1000)
model.fit(inputs[locs[0], :, locs[1], locs[2]].numpy(), torch.masked_select(labels, masks).numpy())

# Pointwise accuracy on train
acc = model.score(inputs[locs[0], :, locs[1], locs[2]].numpy(), torch.masked_select(labels, masks))

inputs_flattened = torch.flatten(inputs.permute(1,0,2,3), -3, -1).T.numpy()
labels_flattened = torch.flatten(labels).T.numpy()
acc_seg = model.score(inputs_flattened, labels_flattened)

metrics['train_acc'] = [acc]
metrics['train_segacc'] = [acc_seg]

print("train: acc", acc)
print("train: seg acc", acc_seg)


inputs, labels, masks = [], [], []
for i, (im, lab, ma) in enumerate(dataloaders['val']):
    if i == 0:
        inputs = im
        labels = lab
        masks = ma
    else:
        inputs = torch.cat((inputs, im), 0)
        labels = torch.cat((labels, lab), 0)
        masks = torch.cat((masks, ma), 0)

locs = np.where(masks == True)

# Pointwise accuracy on train
acc = model.score(inputs[locs[0], :, locs[1], locs[2]].numpy(), torch.masked_select(labels, masks))
inputs_flattened = torch.flatten(inputs.permute(1,0,2,3), -3, -1).T.numpy()
labels_flattened = torch.flatten(labels).T.numpy()
acc_seg = model.score(inputs_flattened, labels_flattened)

print("val: acc", acc)
print("val: seg acc", acc_seg)

metrics['val_acc'] = [acc]
metrics['val_segacc'] = [acc_seg]

results_dir = os.path.join(args.results_dir, f'RF_{params["n_train"]}_{params["label_size"]}_{args.seed}')
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)
