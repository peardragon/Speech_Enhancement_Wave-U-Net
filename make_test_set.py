from model import Model
import librosa
import torch
import glob
from test import test

from tqdm import tqdm

def get_ckpt(path):
    total_list = []
    for x in ['?', '??', '???']:
        ptlist = glob.glob(f"./{path}/ckpt_{x}.pt")
        if(len(ptlist)>0):
            total_list.append(ptlist)

    return total_list[-1][-1]

def get_ckpts(path):
    total_list = []
    for x in ['?', '??', '???']:
        ptlist = glob.glob(f"./{path}/ckpt_{x}.pt")
        if(len(ptlist)>0):
            total_list.append(ptlist)

    return total_list

if __name__ == "__main__":
    # PATH
    PATH = "pt-checkpoints_2021.11.22-04-38"

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
        res, length = test(model, orgin_data)

        librosa.output.write_wav(NOISY_OUTPUT_PATH + file_name, orgin_data[:length], sr=22050)
        librosa.output.write_wav(RESULTS_OUTPUT_PATH + file_name, res, sr=22050)