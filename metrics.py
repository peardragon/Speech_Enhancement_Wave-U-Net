from pesq import pesq as PESQ
from scipy.io import wavfile
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import librosa
from make_test_set import get_ckpts
from model import Model
from test import test
from tqdm import tqdm
import torch
import torchaudio
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy

# ref : clean one  / deg : noisy one
# https://github.com/jdavibedoya/SE_Wave-U-Net/blob/d54aa19245cbe40de10ab2c90486043aad265bb3/Metrics.py

def Eval(ref, deg, sr):
    # Compute the SSNR
    ref_wav, ref_sr = librosa.load(ref, sr=sr, mono=True)
    deg_wav, deg_sr = librosa.load(deg, sr=sr, mono=True)

    if np.abs(len(ref_wav) - len(deg_wav)) < 10:  # tolerate up to 10 samples difference
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = ref_wav[:min_len]
        deg_wav = deg_wav[:min_len]

    assert (len(ref_wav) == len(deg_wav) and ref_sr == deg_sr)
    segsnr_mean = SSNR(ref_wav, deg_wav, ref_sr)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_sr = 16000
    ref_wav, ref_sr = librosa.load(ref, sr=pesq_sr, mono=True)
    deg_wav, deg_sr = librosa.load(deg, sr=pesq_sr, mono=True)

    if np.abs(len(ref_wav) - len(deg_wav)) < 10:  # tolerate up to 10 samples difference
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = ref_wav[:min_len]
        deg_wav = deg_wav[:min_len]

    assert (len(ref_wav) == len(deg_wav) and ref_sr == deg_sr)
    pesq = PESQ(pesq_sr, ref_wav, deg_wav, mode='wb')

    return pesq, segSNR, len(ref_wav)

def SSNR(ref_wav, deg_wav, srate=44100, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [P. C. Loizou, Evaluation of Objective Quality Measures for Speech Enhancement]
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    # Scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + 10e-20))

    # Global variables
    winlength = int(np.round(30 * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    MIN_SNR = -10
    MAX_SNR = 35

    # For each frame, calculate SSNR
    num_frames = int(clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) Get the frames for the test and ref speech
        # Apply Hanning Window
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps)+ eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return segmental_snr

def librosa_waveform(wav, title):
    fig = plt.figure()
    librosa.display.waveplot(wav)
    plt.title(title)

    return fig


def librosa_spectrogram(wav, title, scale='log'):
    D = librosa.stft(wav)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # return shape (1+n_fft/2, n_frames)
    fig = plt.figure()
    librosa.display.specshow(S_db, y_axis=scale, x_axis='time')
    plt.colorbar(format="%+2.f dB")
    plt.title(title)

    return fig


if __name__ == "__main__":

    tb = SummaryWriter("./runs/metrics")
    # sample data
    PATH = "pt-checkpoints_2021.11.22-04-38"
    # p232_023.wav
    models = get_ckpts(PATH)
    models = np.array(sum(models, []))
    NOISY_PATH = "./testset/noisy/"
    CLEAN_PATH = "./testset/clean/"
    OUTPUT_PATH = "./temp/"

    sample_audio = "p232_023.wav"

    TEST_PATH = NOISY_PATH + sample_audio
    CLEAN_PATH += sample_audio

    origin_data, _ = librosa.load(TEST_PATH, sr=22050, mono=True, res_type='kaiser_fast', offset=0.0, duration=None)
    clean_data, _ = librosa.load(CLEAN_PATH, sr=22050, mono=True, res_type='kaiser_fast', offset=0.0, duration=None)
    model = Model()

    glob_step = 0
    for modelckpt in tqdm(models):
        model.load_state_dict(torch.load(modelckpt))
        model.eval()
        idx = modelckpt[modelckpt.rfind("_")+1:-3]
        # print(f"Current Model : {idx}")
        curr_output, length = test(model, origin_data)
        curr_clean = clean_data[:length]
        curr_origin = origin_data[:length]

        maxv = np.iinfo(np.int16).max
        scipy.io.wavfile.write(OUTPUT_PATH + "output.wav", 22050, (curr_output * maxv).astype(np.int16))
        scipy.io.wavfile.write(OUTPUT_PATH + "origin.wav", 22050, (curr_origin * maxv).astype(np.int16))
        scipy.io.wavfile.write(OUTPUT_PATH + "clean.wav", 22050, (curr_clean * maxv).astype(np.int16))

        # librosa.output.write_wav(OUTPUT_PATH + "output.wav", curr_output, sr=22050)
        # librosa.output.write_wav(OUTPUT_PATH + "origin.wav", curr_origin, sr=22050)
        # librosa.output.write_wav(OUTPUT_PATH + "clean.wav", curr_clean, sr=22050)

        if glob_step == 0:
            clean_audio, sr = torchaudio.load(OUTPUT_PATH + "clean.wav")
            tb.add_audio("Clean", clean_audio, global_step=glob_step, sample_rate=sr)

            noisy_audio, sr = torchaudio.load(OUTPUT_PATH + "origin.wav")
            tb.add_audio("Noisy", noisy_audio, global_step=glob_step, sample_rate=sr)

        output_audio, sr = torchaudio.load(OUTPUT_PATH + "output.wav")
        tb.add_audio(f"Output {glob_step}", output_audio, global_step=glob_step, sample_rate=sr)

        # ref : clean one  / deg : noisy one
        pesq, segSNR, _ = Eval(ref=OUTPUT_PATH + "output.wav", deg=OUTPUT_PATH + "origin.wav", sr=22050)

        clean_waveform = librosa_waveform(curr_clean, "Clean")
        noisy_waveform = librosa_waveform(curr_origin, "Noisy")
        output_waveform = librosa_waveform(curr_output, "Output")

        if glob_step == 0:
            tb.add_figure(f"Clean Waveform", clean_waveform, global_step=glob_step)
            tb.add_figure(f"Noisy Waveform", noisy_waveform, global_step=glob_step)
        tb.add_figure(f"Output Waveform", output_waveform, global_step=glob_step)

        clean_spec = librosa_spectrogram(curr_clean, "Clean")
        noisy_spec = librosa_spectrogram(curr_origin, "Noisy")
        output_spec = librosa_spectrogram(curr_output, "Output")

        if glob_step == 0:
            tb.add_figure(f"Clean Spectrogram", clean_spec, global_step=glob_step)
            tb.add_figure(f"Noisy Spectrogram", noisy_spec, global_step=glob_step)
        tb.add_figure(f"Output Spectrogram", output_spec, global_step=glob_step)

        tb.add_scalar("PESQ", pesq, global_step=glob_step)
        tb.add_scalar("SSNR", segSNR, global_step=glob_step)

        glob_step += 1






