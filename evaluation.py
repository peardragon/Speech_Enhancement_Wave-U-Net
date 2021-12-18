import torch
import numpy as np
import librosa
import matplotlib
from model import Model
from dataloader import Dataset
from torch.utils.data import DataLoader

SIZE = 16384


def speech_enhancement(model, test_input):
    """
    With model and noisy input, output the enhancement array
    model : input model path, string.
    test_input : origin noisy input, array. output of function librosa.load().
    """
    data = np.reshape(test_input, (1, 1, -1))
    test_input = data.astype(np.float32)
    # print(test_input.shape)
    num_sampling = np.floor(test_input.shape[2]/16384)
    samples = [test_input[:,:,i*16384:(i+1)*16384] for i in range(int(num_sampling))]
    output_tot = []
    for sample in samples:
        sample = torch.tensor(sample)
        output = model(sample)
        output = output.detach().numpy()
        output_tot.append(output)
    output_tot = np.array(output_tot).flatten()
    output_len = len(output_tot)
    return output_tot, output_len


if __name__ == "__main__":
    PATH = "./wave_u_net_checkpoints/ckpt_72.pt"
    TEST_PATH = "./testset/noisy/p232_023.wav"
    CLEAN_PATH = './testset/clean/p232_023.wav'
    OUTPUT_PATH = "./test_res/"
    model = Model()
    model.to("cpu")
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # dataset = Dataset()
    # BATCH_SIZE = 4
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # random_mixture, _ = next(iter(dataloader))
    # print(random_mixture)

    orgin_data, _ = librosa.load(TEST_PATH, sr=22050, mono=True, res_type='kaiser_fast', offset=0.0, duration=None)
    clean, _ = librosa.load(CLEAN_PATH, sr=22050, mono=True, res_type='kaiser_fast', offset=0.0, duration=None)

    res, length = speech_enhancement(model, orgin_data)
    librosa.output.write_wav(OUTPUT_PATH+"origin.wav", orgin_data[:length], sr=22050)
    librosa.output.write_wav(OUTPUT_PATH+"res.wav", res, sr=22050)
    librosa.output.write_wav(OUTPUT_PATH+"clean.wav", clean[:length], sr=22050)
    print("Done")

