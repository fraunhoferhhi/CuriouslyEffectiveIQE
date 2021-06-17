import torch
import torchvision.transforms
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import PIL.Image as Image
from skimage.util import view_as_blocks
from tqdm import tqdm
import os
import argparse


class FeatureExtraction(Dataset):

    def __init__(self, path_csv, path_codebook, max_blocks, block_size, device, path_zca, use_pil_convert = True, use_pytorch=True):

        self.use_pytorch = use_pytorch
        self.datasheet = pd.read_csv(path_csv)
        self.codebook = np.load(path_codebook)

        if self.use_pytorch:
            self.codebook = torch.Tensor(self.codebook).to(device)

        self.max_blocks = max_blocks
        self.block_size = block_size
        self.device=device
        self.path_zca = path_zca
        if self.path_zca:
            mp = loadmat(args.path_zca)
            self.zca_M = mp['M']
            self.zca_P = mp['P']
        else:
            self.zca_M = None
            self.zca_P = None

        self.eps = 10.
        self.use_pil_convert = use_pil_convert


        if not use_pil_convert:
            self.eps = 0.00001
            self.transform_ToTensor = torchvision.transforms.ToTensor()
            self.transform_Grayscale = torchvision.transforms.Grayscale()

    def __len__(self):
        return self.datasheet.shape[0]

    def __getitem__(self, item):

        """Loads an image from the dataset, extacts patches and applies encoding."""

        row = self.datasheet.loc[item]

        if self.use_pil_convert:
            img = np.atleast_3d(np.array(Image.open(row.path_dist).convert("L")).astype(np.float32))
        else:
            img = Image.open(row.path_dist)
            img = self.transform_ToTensor(img)
            img = self.transform_Grayscale(img)
            img = img.permute(1,2,0).numpy()

        blocks = self.image2blocks(img)
        blocks = blocks.reshape(1, blocks.shape[0], -1)
        blocks = blocks.astype(np.float32)

        blocks = (blocks - np.expand_dims(blocks.mean(axis=2), 2)) / \
                 np.expand_dims(np.sqrt(blocks.var(axis=2) + self.eps), 2)

        if self.path_zca:
            blocks = (blocks - self.zca_M).dot(self.zca_P)

        beta = self.encode(blocks, use_pytorch=self.use_pytorch)

        return row.append(pd.Series({"beta": beta}, dtype=object))


    def encode(self, X, use_pytorch=True):

        """
        Encode features with codebook according to CORNIA paper.

        :param X: np.ndarray of shape (#images, patches_per_image, pixels_per_patch), data to be encoded
        :param pytorch: Whether to use pytorch for feature extraction. Using pytorch can lead to faster performance but
                        will likely lead to results that do not match the results in the paper exactly.
        :return: np.ndarray of shape (#images, #codes in codebook)
        """

        if use_pytorch:
            X = torch.Tensor(X).to(self.device)
            z = X.matmul(self.codebook)
            zpos = torch.max(torch.clamp(z, min=0), dim=1)[0] # shape: (1, 10000)
            zneg = torch.max(torch.clamp(-z, min=0), dim=1)[0] # shape: (1, 10000)
            z = torch.cat([zpos, zneg], 1) # shape: (1, 20000)
            return z.cpu().numpy()

        else:

            # X: [#images, patches_per_img, d]
            z = np.matmul(X, self.codebook)
            # z: [num_images, patches_per_img, K]
            zpos = np.maximum(z, 0) # shape: (1, 6480, 10000)
            zneg = np.maximum(-z, 0) # shape: (1, 6480, 10000)
            z = np.dstack([zpos, zneg]) # shape: (1, 6480, 20000)
            # z: [#images, patches_per_img, 2K]
            z=np.max(z, axis=1) # shape: (1, 20000)
            return z

    def image2blocks(self, img):
        """
        Randomly samples self.max_patches_per_image blocks of size self.block_size from an image.

        :param img: np.ndarray, image as shape (h,w,c)
        :param order: np.ndarray, len(np.ndarray) should match the maximum number of images that could be sampled from
               this image. (This is not the same as self.max_patches_per_image, which is the number of patches that will
               be sampled but which may be lower than the maximum possible number.) The order is a random permutation
               of the blocks and needed to sample spatially co-located patches from distorted and associated reference
               image. In this case sampling from the first image (e.g. the reference) procudes the order to be used for
               sampling from the second image (e.g. the distorted image).

        :return: blocks, np.ndarray of shape (self.max_patches_per_image, self.block_size[0], self.block_size[1],
                 self.block_size[2]) corresponding to sampled blocks.

        :return: order: np.ndarray random permutation of image blocks used to sample blocks, see param order for details.
        """
        H, W, C = img.shape
        h, w, c = self.block_size
        # reduce spatial dimension to a multiple of spatial blocksize
        img = img[:(H // h) * h, :(W // w) * w]
        blocks = view_as_blocks(img, block_shape=self.block_size)
        n, m, _, h, w, c = blocks.shape
        blocks = blocks.reshape(-1, h, w, c)

        order = np.random.permutation(n * m)
        # apply order to blocks and then select the first max_patches_per_image blocks
        blocks = blocks[order]
        # select only max_patches_per_image blocks
        blocks = blocks[:self.max_blocks]

        return blocks


def collate(batch):
    return pd.concat(batch, axis=1).T


def save_features(df, path, overwrite=False):
    tmp_path = path
    if not overwrite:
        k = 0
        while os.path.exists(tmp_path):
            tmp_path = path.replace("_with_beta.pkl", "_with_beta{}.pkl".format(k))
            k += 1

    df.to_pickle(tmp_path)


def main(args):

    # set random seed for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # select cpu or gpu (only relevant when using pytorch option)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up dataset and dataloader
    dataset = FeatureExtraction(path_csv=args.path_csv, max_blocks=args.max_blocks,
                                block_size=args.block_size,
                                device=device,
                                path_codebook=args.path_codebook,
                                path_zca=args.path_zca,
                                use_pil_convert=args.use_pil_convert,
                                use_pytorch=args.use_pytorch)

    dataloader = DataLoader(dataset, collate_fn=collate, batch_size=10)

    # extracted features will be stored in a pandas DataFrame
    results = pd.DataFrame()

    # iterate over all images
    for i, batch in enumerate(tqdm(dataloader)):
        results = results.append(batch, ignore_index=True)

    # save codebook name used for feature extraction
    results.loc[:, "codebook"] = args.name

    # Save results; if there are existing results, append the new results to those
    if os.path.exists(args.path_out):
        _results = pd.read_pickle(args.path_out)
        results = _results.append(results, ignore_index=True)

    save_features(results, args.path_out, overwrite=True)

if __name__ == "__main__":

    def str2bool(v):
        # stolen from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=123,
                        help="Random seed for reproducibility.")

    parser.add_argument("--block_size", type=int, default=7,
                        help="Patch size. (Single number is used for height and width.)")

    parser.add_argument("--max_blocks", type=int, default=10000,
                        help="Maximum number of patches to extract from single image.")

    parser.add_argument("--path_csv", type=str, required=True,
                        help="Path to the dataset (.csv file).")

    parser.add_argument("--path_codebook", type=str, required=True,
                        help="Path to the codebook (.npy file))")

    parser.add_argument("--name", type=str, required=True,
                        help="Name of codebook model.")

    parser.add_argument("--path_zca", type=str2bool, default=False,
                        help="Path to .mat file which contains zca parameters. Only used for CORNIA model.")

    parser.add_argument("--path_out", type=str, default="features.pkl",
                        help="Path under which to save extracted features")

    parser.add_argument("--use_pil_convert", type=str2bool, default=False,
                        help="Whether to PIL.Image.convert() for color conversion from RGB to grayscale. "
                             "If False, use pytorch for color conversion.")

    parser.add_argument("--use_pytorch", type=str2bool, default=True,
                        help="Pytorch allows for gpu acceleration but results might slightly deviate from the paper.")

    parser.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers. Has to be 0 if you are "
                                                                   "using the pytorch version.")

    args = parser.parse_args()
    if args.path_out is None:
        args.path_out = os.path.join(".", args.path_csv.split("/")[-1].replace(".csv", "_with_beta.pkl"))
    args.block_size = (args.block_size, args.block_size, 1)

    main(args)