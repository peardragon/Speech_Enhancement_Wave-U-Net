from model import Model
import librosa
import torch
import glob
from evaluation import speech_enhancement
import scipy
from scipy.io import wavfile
import numpy as np

from tqdm import tqdm

def get_ckpt(path):
    """
    get model final version path
    path: parent directory of model checkpoints, string
    """
    total_list = []
    for x in ['?', '??', '???']:
        ptlist = glob.glob(f"{path}/ckpt_{x}.pt")
        if(len(ptlist)>0):
            total_list.append(ptlist)

    return total_list[-1][-1]

def get_ckpts(path):
    """
    get model versions paths as list
    path: parent directory of model checkpoints, string
    """
    total_list = []
    for x in ['?', '??', '???']:
        ptlist = glob.glob(f"{path}/ckpt_{x}.pt")
        if(len(ptlist)>0):
            total_list.append(ptlist)

    return total_list

if __name__ == "__main__":
    # PATH
    PATH = "./wave_u_net_ckpt"

    testset_list = [line.rstrip('\n') for line in open("testset/testset_list.txt", "r")]
    model = Model()
    path = get_ckpt(PATH)
    print(path)
    model.load_state_dict(torch.load(path))
    model.eval()

    NOISY_OUTPUT_PATH = "./testset/model_results/noisy/"
    RESULTS_OUTPUT_PATH = "./testset/model_results/clean/"

    for idx in tqdm(range(len(testset_list))):
        mixture_path, clean_path = testset_list[idx].split(" ")
        file_name = mixture_path[-12:]
        orgin_data, _ = librosa.load(mixture_path, sr=22050, mono=True,
                                     res_type='kaiser_fast', offset=0.0, duration=None)
        res, length = speech_enhancement(model, orgin_data)

        maxv = np.iinfo(np.int16).max
        scipy.io.wavfile.write(NOISY_OUTPUT_PATH, 22050, (orgin_data[:length] * maxv).astype(np.int16))
        scipy.io.wavfile.write(RESULTS_OUTPUT_PATH, 22050, (res * maxv).astype(np.int16))