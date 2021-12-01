import glob
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
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

    x = glob.glob(conf["hpcp"]["mp3_dir"] + "/*/*.mp3")
    os.mkdir(conf["hpcp"]["output_dir"])
    for i in range(10):
        for j in range(10):
            os.mkdir(conf["hpcp"]["output_dir"] + "/{}{}/".format(i, j))

    def op(name):
        target_path = name.replace(conf["hpcp"]["mp3_dir"].split("/")[-2], 
                                   conf["hpcp"]["output_dir"].split("/")[-2]).replace("mp3", "npy")
        
        if os.path.exists(target_path):
            print("Skipping {}".format(target_path))
        else:
            audio = estd.MonoLoader(filename=name, sampleRate=44100)()
            hpcp = hpcpgram(audio, sampleRate=44100)
            hpcp = torch.tensor(hpcp)
            hpcp = hpcp.unsqueeze(0).unsqueeze(0)
            if conf["hpcp"]["pool_size"] > 1:
                hpcp = nn.AvgPool2d((conf["hpcp"]["pool_size"], 1), stride=(conf["hpcp"]["pool_size"], 1))(hpcp).squeeze()
            hpcp = hpcp.numpy()[:conf["hpcp"]["cutoff_size"], :]
            assert hpcp.shape[-1] == 12
            np.save(target_path, hpcp)

    pool = ThreadPool(12)
    pbar = tqdm(total=len(x))

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        pool.apply_async(op, args=(x[i],), callback=update)
    pool.close()
    pool.join()
        
