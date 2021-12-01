import glob
from nnAudio import Spectrogram
import librosa
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import os
from multiprocessing.dummy import Pool as ThreadPool
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, required=True,
                        help="config file")

    args = parser.parse_args()
    if not os.path.exists(args.conf):
        print("Config file not found. Ending...")
    
    with open(args.conf, "r+") as f:
        conf = json.load(f)

    x = glob.glob(conf["melspec"]["mp3_dir"] + "/*/*.mp3")
    melspec_op = Spectrogram.MelSpectrogram(sr=44100).cuda()

    os.mkdir(conf["melspec"]["output_dir"])
    for i in range(10):
        for j in range(10):
            os.mkdir(conf["melspec"]["output_dir"] + "/{}{}/".format(i, j))

    def op(name):
        target_path = name.replace(conf["melspec"]["mp3_dir"].split("/")[-2], 
                                   conf["melspec"]["output_dir"].split("/")[-2]).replace("mp3", "npy")
        if os.path.exists(target_path):
            print("Skipping {}".format(target_path))
        else:
            audio, sr = librosa.load(name, sr=44100)
            audio = torch.tensor(audio).cuda()
            melspec = melspec_op(audio)
            melspec = melspec.unsqueeze(0)
            melspec = nn.AvgPool2d((1, conf["melspec"]["pool_size"]), stride=(1, conf["melspec"]["pool_size"]))(melspec)
            melspec = melspec.cpu().detach().numpy().squeeze()[:, :conf["melspec"]["cutoff_size"]]
            assert melspec.shape[0] == 128
            assert melspec.shape[1] > 0
            np.save(target_path, melspec)

    pool = ThreadPool(12)
    pbar = tqdm(total=len(x))

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        pool.apply_async(op, args=(x[i],), callback=update)
    pool.close()
    pool.join()