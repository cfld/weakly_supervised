from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import glob
import os
import numpy as np


def cdl_to_binary(cdl):
    #print(((cdl <= 60) | (cdl >= 196)) | ((cdl >= 66) & (cdl <= 77)))
    return (((cdl <= 60) | (cdl >= 196)) | ((cdl >= 66) & (cdl <= 77)))

class MaskedTileDataset(Dataset):

    def __init__(self, tile_dir, tile_files, transform=None, n_samples=None):

        self.tile_dir = tile_dir
        self.tile_files = tile_files
        self.transform = transform
        self.n_samples = n_samples

        # Generate a set of mask points which remain fixed
        self.masks_pix = {i: np.random.randint(1,49,2) for i in range(len(self.tile_files))}
        print(self.masks_pix, "MASKS PIX")

    def __len__(self):
        if self.n_samples:
            return self.n_samples
        else:
            return len(self.tile_files)

    def __getitem__(self, idx):
        tile = np.load(os.path.join(self.tile_dir, self.tile_files[idx]))
        tile = np.nan_to_num(tile)
        tile = np.moveaxis(tile, -1, 0)

        mask = np.zeros((tile.shape[1], tile.shape[2]), dtype=bool)
        mask[self.masks_pix[idx][0],self.masks_pix[idx][1]] = True
        mask = np.expand_dims(mask, axis=0)
        tile = np.concatenate([tile, mask], axis=0)  # attach mask to tile to ensure same transformations are applied

        if self.transform:
            tile = self.transform(tile)
        features = tile[:7, :, :]

        label = tile[-2, :, :] * 10000
        label = cdl_to_binary(label)
        label = label.float()

        mask = tile[-1, :, :]
        mask = mask.bool()

        return features, label, mask


class MaskedTileDataset_naip(Dataset):

    def __init__(self, tile_dir, tile_files, transform=None, n_samples=None, im_size=50):

        self.tile_dir = tile_dir
        self.tile_files = tile_files
        self.transform = transform
        self.n_samples = n_samples
        self.im_size   = im_size

        # Generate a set of mask points which remain fixed
        self.masks_pix = {i: np.random.randint(0,im_size,2) for i in range(len(self.tile_files))}

    def __len__(self):
        if self.n_samples:
            return self.n_samples
        else:
            return len(self.tile_files)

    def __getitem__(self, idx):
        tile = np.load(os.path.join(self.tile_dir, self.tile_files[idx]))[:, 0:self.im_size, 0:self.im_size]
        tile = np.nan_to_num(tile)

        mask = np.zeros((tile.shape[1], tile.shape[2]), dtype=bool)
        mask[self.masks_pix[idx][0],self.masks_pix[idx][1]] = True
        mask = np.expand_dims(mask, axis=0)
        tile = np.concatenate([tile, mask], axis=0)  # attach mask to tile to ensure same transformations are applied

        if self.transform:
            tile = self.transform(tile)
        features = (tile[:4, :, :] - NAIP_BAND_STATS['mean'].transpose(2,0,1))/NAIP_BAND_STATS['std'].transpose(2,0,1)
        features = features.float()

        label = tile[4, :, :]
        label = cdl_to_binary(label)
        label = label.float()

        mask = tile[-1, :, :]
        mask = mask.bool()

        return features, label, mask



class MaskedTileDataset_sentinel(Dataset):

    def __init__(self, tile_dir, tile_files, transform=None, n_samples=None, im_size=50):

        self.tile_dir = tile_dir
        self.tile_files = tile_files
        self.transform = transform
        self.n_samples = n_samples
        self.im_size   = im_size

        # Generate a set of mask points which remain fixed
        self.masks_pix = {i: np.random.randint(0,im_size,2) for i in range(len(self.tile_files))}

    def __len__(self):
        if self.n_samples:
            return self.n_samples
        else:
            return len(self.tile_files)

    def __getitem__(self, idx):
        tile = np.load(os.path.join(self.tile_dir, self.tile_files[idx]))[:, 0:self.im_size, 0:self.im_size]
        tile = np.nan_to_num(tile).astype(np.float32) / 10_000

        mask = np.zeros((tile.shape[1], tile.shape[2]), dtype=bool)
        mask[self.masks_pix[idx][0],self.masks_pix[idx][1]] = True
        mask = np.expand_dims(mask, axis=0)
        tile = np.concatenate([tile, mask], axis=0)  # attach mask to tile to ensure same transformations are applied

        if self.transform:
            tile = self.transform(tile)
        features = (tile[:12, :, :] - SENTINEL_BAND_STATS['mean'].transpose(2,0,1))/SENTINEL_BAND_STATS['std'].transpose(2,0,1)
        features = features.float()

        label = tile[13, :, :] * 10000
        label = cdl_to_binary(label)
        label = label.float()

        mask = tile[-1, :, :]
        mask = mask.bool()

        return features, label, mask

class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, tile):
        # randomly flip
        if np.random.rand() < 0.5: tile = np.flip(tile, axis=2).copy()
        if np.random.rand() < 0.5: tile = np.flip(tile, axis=1).copy()
        
        # randomly rotate
        rotations = np.random.choice([0,1,2,3])
        if rotations > 0: tile = np.rot90(tile, k=rotations, axes=(1,2)).copy()
        
        return tile


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, tile):
        tile = torch.from_numpy(tile).float()
        return tile


NAIP_BAND_STATS = {
    'mean' : np.array([0.38194386, 0.38695849, 0.35312921, 0.45349037])[None, None],
    'std'  : np.array([0.21740159, 0.18325207, 0.15651401, 0.20699527])[None, None],
}

SENTINEL_BAND_STATS = {
    'mean' : np.array([1.08158484e-01, 1.21479766e-01, 1.45487537e-01, 1.58012632e-01, 1.94156398e-01, 2.59219257e-01,
                       2.83195732e-01, 2.96798923e-01, 3.01822935e-01, 3.08726458e-01, 2.37724304e-01, 1.72824851e-01])[None, None],
    'std'  : np.array([2.00349529e-01, 2.06218237e-01, 1.99808794e-01, 2.05981393e-01, 2.00533060e-01, 1.82050607e-01,
                       1.76569472e-01, 1.80955308e-01, 1.68494856e-01, 1.80597534e-01, 1.15451671e-01, 1.06993609e-01])[None, None],
}


def masked_tile_dataloader(tile_dir, tile_files, augment=True, batch_size=4, shuffle=True, num_workers=4, n_samples=None, im_size=50):
    """
    Returns a dataloader with Landsat tiles.
    """
    transform_list = []
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)

    dataset = MaskedTileDataset_sentinel(tile_dir, tile_files, transform=transform, n_samples=n_samples, im_size=im_size)
    #dataset = MaskedTileDataset_naip(tile_dir, tile_files, transform=transform, n_samples=n_samples, im_size=im_size)
    #dataset = MaskedTileDataset(tile_dir, tile_files, transform=transform, n_samples=n_samples)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
