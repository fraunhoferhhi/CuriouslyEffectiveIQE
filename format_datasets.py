import argparse
import os
import glob
import zipfile
import rarfile
import sys
import shutil
from shutil import copyfile

import numpy as np
import PIL.Image as Image
import pandas as pd

from scipy.io import loadmat
from skimage.metrics import mean_squared_error
from tqdm import tqdm



def format_csiq(args):

    """
    To format the CSIQ dataset, download the following files:
        1) src_imgs.zip
        2) dst_imgs.zip
        3) csiq.DMOS.xlsx
    from http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23 .

    The formatted dataset will be summarized in <args.data_dir>/csiq.csv.
    """

    if not os.path.exists(os.path.join(args.data_dir, "src_imgs.zip")):
        print("You need to download 'src_imgs.zip' from the dataset's websites.")
        sys.exit()

    if not os.path.exists(os.path.join(args.data_dir, "dst_imgs.zip")):
        print("You need to download 'dst_imgs.zip' from the dataset's websites.")
        sys.exit()

    if not os.path.exists(os.path.join(args.data_dir, "csiq.DMOS.xlsx")):
        print("You need to download 'csiq.DMOS.xlsx' from the dataset's websites.")
        sys.exit()

    def get_dist_type(x):
        if x == "jpeg":
            return "jpeg", "jpeg"
        elif x == "jpeg 2000":
            return "jpeg2000", "jp2k"
        elif x == "blur":
            return "blur", "gblur"
        elif x == "fnoise":
            return "fnoise", "fnoise"
        elif x == "noise":
            return "awgn", "awgn"
        elif x == "contrast":
            return "contrast", "contrast"
        else:
            raise ValueError("Unknown distortion type: {}".format(x))

    with zipfile.ZipFile(os.path.join(args.data_dir, "src_imgs.zip"), 'r') as zip_ref:
        print("Extracting src_imgs.zip")
        zip_ref.extractall(os.path.join(args.data_dir, "src_imgs"))

    with zipfile.ZipFile(os.path.join(args.data_dir, "dst_imgs.zip"), 'r') as zip_ref:
        print("Extracting dst_imgs.zip")
        zip_ref.extractall(os.path.join(args.data_dir, "dst_imgs"))

    scores = pd.read_excel(os.path.join(args.data_dir, "csiq.DMOS.xlsx"),
                           sheet_name="all_by_image", header=3, usecols=[3, 5, 6, 8])

    for path in glob.glob(os.path.join(args.data_dir, "src_imgs", "*")):
        os.rename(path, path.lower())

    for path in glob.glob(os.path.join(args.data_dir, "dst_imgs", "*", "*")):
        os.rename(path, path.lower())

    if not os.path.exists(os.path.join(args.data_dir, "reference")):
        os.makedirs(os.path.join(args.data_dir, "reference"))

    db = pd.DataFrame(columns=["dataset", "refname", "distortion",
                               "height", "width", "fps",
                               "path_ref", "path_dist", "mse", "quality"])

    for i, row in tqdm(scores.iterrows()):

        distType_file, distType_new = get_dist_type(row.dst_type)

        # original name of reference image
        img_name_ref = "{}.png".format(row.image)
        # new path for reference image
        path_img_ref_new = glob.glob(
            os.path.join(args.data_dir, "reference", "*_{}_*".format(row.image)))

        if len(path_img_ref_new) == 0:
            # reference image has not been copied to new directory yet
            img_ref = np.array(
                Image.open(os.path.join(args.data_dir, "src_imgs", img_name_ref)
                           ).convert("L"))

            h, w = img_ref.shape[0:2]

            img_name_ref_new = "csiq_{}_{}x{}_1_ref.png".format(
                row.image, h, w)

            os.rename(os.path.join(args.data_dir, "src_imgs", img_name_ref),
                      os.path.join(args.data_dir, "reference", img_name_ref_new))

        elif len(path_img_ref_new) == 1:
            # reference image has already been copied
            img_ref = np.array(Image.open(path_img_ref_new[0]
                                          ).convert("L"))

            h, w = img_ref.shape[0:2]

            img_name_ref_new = "csiq_{}_{}x{}_1_ref.png".format(
                row.image, h, w)
        else:
            raise ValueError("Multiple reference images found for glob " \
                             "expression '*_{}_*'".format(row.image))

        img_name_dist = "{}.{}.{}.png".format(
            row.image, distType_file, row.dst_lev)

        img_dist = np.array(Image.open(os.path.join(
            args.data_dir, "dst_imgs", distType_file, img_name_dist)
        ).convert("L"))

        img_name_dist_new = "csiq_{}_{}x{}_1_{}_{:03.2f}.png".format(
            row.image, h, w, distType_new, row.dmos)

        # create directory for this distortion type
        if not os.path.exists(os.path.join(args.data_dir, distType_new)):
            os.makedirs(os.path.join(args.data_dir, distType_new))

        # copy distorted image
        os.rename(
            os.path.join(args.data_dir, "dst_imgs", distType_file, img_name_dist),
            os.path.join(args.data_dir, distType_new, img_name_dist_new))

        # compute mse
        mse = mean_squared_error(img_ref, img_dist)

        rowIdx = db.shape[0]
        db.loc[rowIdx, "dataset"] = "csiq"
        db.loc[rowIdx, "refname"] = row.image
        db.loc[rowIdx, "distortion"] = distType_new
        db.loc[rowIdx, "height"] = h
        db.loc[rowIdx, "width"] = w
        db.loc[rowIdx, "fps"] = 1
        db.loc[rowIdx, "path_ref"] = os.path.join(args.data_dir, "reference", img_name_ref_new)

        db.loc[rowIdx, "path_dist"] = os.path.join(args.data_dir, distType_new, img_name_dist_new)

        db.loc[rowIdx, "mse"] = mse
        db.loc[rowIdx, "quality"] = row.dmos

    # normalize quality scores
    q_range = db.quality.max() - db.quality.min()
    db.loc[:, "q_norm"] = 1 - (db.quality - db.quality.min()) / q_range

    path_csv = os.path.join(args.data_dir, "csiq.csv")

    db.to_csv(path_csv)

    shutil.rmtree(os.path.join(args.data_dir, "src_imgs"))
    shutil.rmtree(os.path.join(args.data_dir, "dist_imgs"))

    return path_csv


def format_tid2013(args):

    """
    Format the TID2013 dataset. You will need to first (manually) download the file 'tid2013.rar' from
    http://www.ponomarenko.info/tid2013.htm . The formatted dataset will be summarized in <args.data_dir>/tid2013.csv.
    """
    if not os.path.exists(os.path.join(args.data_dir, "tid2013.rar")):
        print("You need to download 'tid2013.rar' from the dataset's websites.")
        sys.exit()

    def index2distortion(idx):

        if idx == 1: return "awgn"
        if idx == 2: return "awgn2"
        if idx == 3: return "scn"
        if idx == 4: return "mn"
        if idx == 5: return "hfn"
        if idx == 6: return "in"
        if idx == 7: return "qn"
        if idx == 8: return "gblur"
        if idx == 9: return "id"
        if idx == 10: return "jpeg"
        if idx == 11: return "jp2k"
        if idx == 12: return "jpegt"
        if idx == 13: return "jp2kt"
        if idx == 14: return "nepn"
        if idx == 15: return "lbdi"
        if idx == 16: return "ms"
        if idx == 17: return "cc"
        if idx == 18: return "ccs"
        if idx == 19: return "mgn"
        if idx == 20: return "cn"
        if idx == 21: return "lcni"
        if idx == 22: return "icqd"
        if idx == 23: return "ca"
        if idx == 24: return "ssr"

        raise ValueError("Unknown distortion index: {}".format(idx))

    assert os.path.exists(os.path.join(args.data_dir, "tid2013.rar"))

    with rarfile.RarFile(os.path.join(args.data_dir, "tid2013.rar")) as rf:
        print("Extracting tid2013.rar")
        rf.extractall(args.data_dir)

    for path in glob.glob(os.path.join(args.data_dir, "reference_images", "*")):
        os.rename(path, path.lower())

    for path in glob.glob(os.path.join(args.data_dir, "distorted_images", "*")):
        os.rename(path, path.lower())

    if not os.path.exists(os.path.join(args.data_dir, "reference")):
        os.makedirs(os.path.join(args.data_dir, "reference"))

    db = pd.DataFrame(columns=["dataset", "refname", "distortion",
                               "height", "width", "fps",
                               "path_ref", "path_dist", "mse", "quality"])

    with open(os.path.join(args.data_dir, "mos_with_names.txt"), "r") as mos_file:

        for i, line in tqdm(enumerate(mos_file)):

            mos, filename = line.rstrip("\n").lower().split(" ")

            mos = float(mos)

            path_img_dist = glob.glob(os.path.join(args.data_dir, "distorted_images", filename))[0]

            name, idx, level = filename.split(".")[0].split("_")
            dist = index2distortion(int(idx))

            path_img_ref = glob.glob(os.path.join(args.data_dir, "reference_images", name + ".bmp"))[0]

            img_ref = np.array(Image.open(path_img_ref).convert("L"))
            img_dist = np.array(Image.open(path_img_dist).convert("L"))

            mse = mean_squared_error(img_ref, img_dist)
            h, w = img_ref.shape

            new_name_ref = "tid2013_{}_{}x{}_1_ref.bmp".format(name, h, w)
            new_name_dist = "tid2013_{}_{}x{}_1_{}_{:03.2f}.bmp".format(name, h, w, dist, mos)

            if not os.path.exists(os.path.join(args.data_dir, "reference", new_name_ref)):
                copyfile(path_img_ref, os.path.join(args.data_dir, "reference", new_name_ref))

            if not os.path.exists(os.path.join(args.data_dir, dist)):
                os.makedirs(os.path.join(args.data_dir, dist))

            os.rename(path_img_dist, os.path.join(args.data_dir, dist, new_name_dist))

            rowIdx = db.shape[0]
            db.loc[rowIdx, "dataset"] = "tid2013"
            db.loc[rowIdx, "refname"] = name
            db.loc[rowIdx, "distortion"] = dist
            db.loc[rowIdx, "height"] = h
            db.loc[rowIdx, "width"] = w
            db.loc[rowIdx, "fps"] = 1
            db.loc[rowIdx, "path_ref"] = os.path.join(args.data_dir, "reference", new_name_ref)

            db.loc[rowIdx, "path_dist"] = os.path.join(args.data_dir, dist, new_name_dist)

            db.loc[rowIdx, "mse"] = mse
            db.loc[rowIdx, "quality"] = mos

        q_range = db.quality.max() - db.quality.min()

        db.loc[:, "q_norm"] = (db.quality - db.quality.min()) / q_range

        path_csv = os.path.join(args.data_dir, "tid2013.csv")

        db.to_csv(path_csv)

        shutil.rmtree(os.path.join(args.data_dir, "distorted_images"))
        shutil.rmtree(os.path.join(args.data_dir, "reference_images"))

        return path_csv

def format_liveiqa(self):

    """
    Format the LIVE IQA dataset.

    Format the LIVE subjective database release 2. You need to first (manually) download the following files from
    https://live.ece.utexas.edu/research/Quality/subjective.htm :
        1) databaserelease2.zip
        2) dmos_realigned.mat

    The formatted dataset will be summarized in <args.data_dir>/liveiqa.csv.
    """

    if not os.path.exists(os.path.join(args.data_dir, "databaserelease2.zip")):
        print("You need to download 'databaserelease2.zip' from the dataset's websites.")
        sys.exit()

    if not os.path.exists(os.path.join(args.data_dir, "dmos_realigned.mat")):
        print("You need to download 'dmos_realigned.mat' from the dataset's websites.")
        sys.exit()

    def index2distortion(idx):
        if idx < 227:
            return "jp2k", 0, "jp2k"
        elif idx < 227 + 233:
            return "jpeg", 227, "jpeg"
        elif idx < 227 + 233 + 174:
            return "wn", 227 + 233, "awgn"
        elif idx < 227 + 233 + 174 + 174:
            return "gblur", 227 + 233 + 174, "gblur"
        else:
            return "fastfading", 227 + 233 + 174 + 174, "fastfading"

    password = input("Please enter password for databaserelease2.zip as obtained from the LIVE lab:\n")

    with zipfile.ZipFile(os.path.join(args.data_dir, "databaserelease2.zip"), 'r') as zf:
        print("Extracting databaserelease2.zip")
        zf.extractall(os.path.join(args.data_dir), pwd = bytes(password, 'utf-8'))

    source = os.path.join(args.data_dir, "databaserelease2")

    dmoses = loadmat(os.path.join(args.data_dir, "dmos_realigned.mat"))
    dmoses = dmoses["dmos_new"].squeeze()
    names = np.hstack(np.hstack(loadmat(os.path.join(source, "refnames_all.mat"))["refnames_all"]))

    db = pd.DataFrame(columns=["dataset", "refname", "distortion",
                               "height", "width", "fps",
                               "path_ref", "path_dist", "mse", "quality"])

    for i, (name, dmos) in tqdm(enumerate(zip(names, dmoses))):

        if dmos == 0:
            # reference image
            continue

        # get distortion type
        dist, offset, newDist = index2distortion(i)

        # load image
        ref_img = Image.open(os.path.join(source, "refimgs", name)).convert("L")
        dist_img = Image.open(os.path.join(source, dist, "img{}.bmp".format(i + 1 - offset))).convert("L")

        # get resolution
        res_x = ref_img.size[0]
        res_y = ref_img.size[1]

        # compute mse
        mse = mean_squared_error(np.array(ref_img), np.array(dist_img))

        # create directory for reference images
        if not os.path.exists(os.path.join(args.data_dir, "reference")):
            os.makedirs(os.path.join(args.data_dir, "reference"))

        # create directory for this distortion type
        if not os.path.exists(os.path.join(args.data_dir, newDist)):
            os.makedirs(os.path.join(args.data_dir, newDist))

        # construct new names
        new_img_name = name.split(".")[0].replace("_", "").lower()

        new_name_ref = "liveiqa_{}_{}x{}_1_ref.bmp".format(new_img_name, res_x, res_y)

        new_name_dst = "liveiqa_{}_{}x{}_1_{}_{:03.2f}.bmp".format(new_img_name, res_x, res_y, newDist, dmos)

        new_path_ref = os.path.join(args.data_dir, "reference", new_name_ref)
        new_path_dist = os.path.join(args.data_dir, newDist, new_name_dst)

        # copy reference image
        if not os.path.exists(os.path.join(args.data_dir, new_path_ref)):
            copyfile(os.path.join(source, "refimgs", name), new_path_ref)

        # copy distorted image
        if not os.path.exists(os.path.join(args.data_dir, new_path_dist)):
            os.rename(os.path.join(source, dist, "img{}.bmp".format(i + 1 - offset)), new_path_dist)

        rowIdx = db.shape[0]
        db.loc[rowIdx, "dataset"] = "liveiqa"
        db.loc[rowIdx, "refname"] = new_img_name
        db.loc[rowIdx, "distortion"] = newDist
        db.loc[rowIdx, "height"] = res_y
        db.loc[rowIdx, "width"] = res_x
        db.loc[rowIdx, "fps"] = 1
        db.loc[rowIdx, "path_ref"] = new_path_ref
        db.loc[rowIdx, "path_dist"] = new_path_dist
        db.loc[rowIdx, "mse"] = mse
        db.loc[rowIdx, "quality"] = dmos

    q_range = db.quality.max() - db.quality.min()

    db.loc[:, "q_norm"] = 1 - (db.quality - db.quality.min()) / q_range

    path_csv = os.path.join(args.data_dir, "liveiqa.csv")

    db.to_csv(path_csv)

    shutil.rmtree(os.path.join(args.data_dir, "databaserelease2"))

    return path_csv



if __name__ == "__main__":

    """
    This script acts as an adapter for the LIVEIQA, TID2013, and CSIQ datasets, 
    to bring them into a consistent data format. To use this script, you first need to (manually) download the respective
    dataset.
    
    ---------------------------------------LIVE subjective database release 2-------------------------------------------
    
    Download the following file from https://live.ece.utexas.edu/research/quality/subjective.htm :
        1) databaserelease2.zip
        
    (To access this zip file you need to obtain the password from the LIVE lab.)
    
    Assuming you placed 'databaserelease2.zip' under ./data/liveiqa/databaserelease2.zip, you can use the following command:
    python format_datasets.py --dataset liveiqa --data_dir ./data/liveiqa 
    
    --------------------------------------------------TID2013----------------------------------------------------------- 
    
    Download the following file from http://www.ponomarenko.info/tid2013.htm :
        1) tid2013.rar
        
    Assuming you placed 'tid2013.rar' under ./data/tid2013/tid2013.rar, you can use the following command:
    python format_datasets.py --dataset tid2013 --data_dir ./data/tid2013
    
    ----------------------------------------------------CSIQ------------------------------------------------------------ 
    
     Download the following files from http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23 :
        1) src_imgs.zip
        2) dst_imgs.zip
        3) csiq.DMOS.xlsx
    
    Assuming you placed these files under ./data/csiq/*, you can use the following command:
    python format_datasets.py --dataset csiq --data_dir ./data/csiq
    """


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

    parser.add_argument("--dataset", type=str, default="csiq", choices=["liveiqa", "tid2013", "csiq"],
                        help="Dataset to format.")

    parser.add_argument("--data_dir", type=str, default="data/csiq",
                        help="Path to directory containing all files of respective dataset.")

    args = parser.parse_args()

    if args.dataset == "tid2013":
        format_tid2013(args)

    if args.dataset == "csiq":
        format_csiq(args)

    if args.dataset == "liveiqa":
        format_liveiqa(args)


