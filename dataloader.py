import os

import librosa
from torch.utils import data
import numpy as np
import glob

# https://github.com/noise-suppression/Wave-U-Net-for-Speech-Enhancement
def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """
    sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end], data_b[start:end]

class Dataset(data.Dataset):
    def __init__(self,
                 size=2000,
                 batch_size=16,
                 sample_length=16384,
                 mode="train"):
        """Construct dataset for training and validation.
        Args:
            dataset (str): *.txt, the path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.
            sample_length(int): The model only supports fixed-length input. Use sample_length to specify the feature size of the input.
            mode(str): If mode is "train", return fixed-length signals. If mode is "validation", return original-length signals.

        Notes:
            dataset list file：
            <noisy_1_path><space><clean_1_path>
            <noisy_2_path><space><clean_2_path>
            ...
            <noisy_n_path><space><clean_n_path>

            e.g.
            /train/noisy/a.wav /train/clean/a.wav
            /train/noisy/b.wav /train/clean/b.wav
            ...

        Return:
            (mixture signals, clean signals, filename)
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open("dataset/dataset_list.txt", "r")]

        assert mode in ("train", "val", "validation"), "Mode must be one of 'train' or 'validation'."

        if mode == "train":
            dataset_list = dataset_list[:-10]
            INIT_LIMIT = size * batch_size
            limit = (int((INIT_LIMIT / batch_size) * 16))
            if limit <= len(dataset_list):
                len_total = len(dataset_list)
                start = np.random.randint(len_total - limit + 1)
                # print(f"Random crop from: {start}")
                end = start + limit
                dataset_list = dataset_list[start:end]
        else:
            dataset_list = dataset_list[-10:]

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mixture_path, clean_path = self.dataset_list[idx].split(" ")
        filename = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture, _ = librosa.load(mixture_path, sr=22050, mono=True, res_type='kaiser_fast', offset=0.0, duration=None)
        clean, _ = librosa.load(clean_path, sr=22050, mono=True, res_type='kaiser_fast', offset=0.0, duration=None)


        # The input of model should be fixed-length in the training.
        mixture, clean = sample_fixed_length_data_aligned(mixture, clean, self.sample_length)
        return mixture.reshape(1, -1), clean.reshape(1, -1)


if __name__=="__main__":
    f = open("dataset/dataset_list.txt", 'w')
    namelist = glob.glob("dataset/noisy/*.wav")
    for name in namelist:
        idx = name[-12:]
        string = f'dataset/noisy/{idx} dataset/clean/{idx}\n'
        f.write(string)
    f.close()

    from torch.utils.data import DataLoader

    train_dataset = Dataset(size=2000, batch_size=32, mode="test")
    dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True)

    real_batch = next(iter(dataloader))
    print(len(dataloader))
    print(real_batch[0].shape)

