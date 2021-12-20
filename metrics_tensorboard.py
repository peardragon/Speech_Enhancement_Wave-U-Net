from metrics import *
import glob


def librosa_waveform(wav, title):
    """
    return waveform plot as matplotlib figure type.

    wav : .wav file directory, string. what .wav file to use, return spectrogram plot as matplotlib figure type.
    title : title for matplotlib, string.
    """
    fig = plt.figure()
    librosa.display.waveplot(wav)
    plt.title(title)

    return fig


def librosa_spectrogram(wav, title, scale='log'):
    """
    return spectrogram plot as matplotlib figure type.

    wav : .wav file directory, string. what .wav file to use, return spectrogram plot as matplotlib figure type.
    title : title for matplotlib, string.
    scale : matplotlib y-axis scale, string. default is log
    """
    D = librosa.stft(wav)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # return shape (1+n_fft/2, n_frames)
    fig = plt.figure()
    librosa.display.specshow(S_db, y_axis=scale, x_axis='time')
    plt.colorbar(format="%+2.f dB")
    plt.title(title)

    return fig


def sample_audio_evaluation_tensorboard(sample_audio, tb_log_dir, model_path, glob_step):
    """
    sample_audio : sample audio directory, string. with this sample audio, make spectrograph image, several metrics, and save these as tensorboard log file
    tb_log_dir : tensorboard log directory, string. where you save tensorboard log file.
    model_path : model checkpoint directroy, string. what model you want to evaluate.
    glob_step : tensorboard global step variable, int. In tensorboard, step is written by this value.
    """
    tb = SummaryWriter(tb_log_dir)

    # Default path setting

    NOISY_PATH = "./testset/noisy/"
    CLEAN_PATH = "./testset/clean/"
    OUTPUT_PATH = "./temp/"

    sample_audio = sample_audio
    TEST_PATH = NOISY_PATH + sample_audio
    CLEAN_PATH += sample_audio

    sample_name = sample_audio[:-4]

    origin_data, _ = librosa.load(TEST_PATH, sr=22050, mono=True, res_type='kaiser_fast', offset=0.0, duration=None)
    clean_data, _ = librosa.load(CLEAN_PATH, sr=22050, mono=True, res_type='kaiser_fast', offset=0.0, duration=None)

    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # print(f"Current Model : {idx}")
    curr_output, length = speech_enhancement(model, origin_data)
    if length == 0:
        return None, None
    curr_clean = clean_data[:length]
    curr_origin = origin_data[:length]

    maxv = np.iinfo(np.int16).max

    # if(sample_name in tensorboard_audio_list):
    scipy.io.wavfile.write(OUTPUT_PATH + "output.wav", 22050, (curr_output * maxv).astype(np.int16))
    scipy.io.wavfile.write(OUTPUT_PATH + "origin.wav", 22050, (curr_origin * maxv).astype(np.int16))
    scipy.io.wavfile.write(OUTPUT_PATH + "clean.wav", 22050, (curr_clean * maxv).astype(np.int16))

    if glob_step == 0:
        clean_audio, sr = torchaudio.load(OUTPUT_PATH + "clean.wav")
        tb.add_audio(f"{sample_name} Clean", clean_audio, global_step=glob_step, sample_rate=sr)

        noisy_audio, sr = torchaudio.load(OUTPUT_PATH + "origin.wav")
        tb.add_audio(f"{sample_name} Noisy", noisy_audio, global_step=glob_step, sample_rate=sr)

    output_audio, sr = torchaudio.load(OUTPUT_PATH + "output.wav")
    tb.add_audio(f"{sample_name} Output {glob_step}", output_audio, global_step=glob_step, sample_rate=sr)

    # ref : clean one  / deg : predict one
    pesq, segSNR, _ = Eval(ref=OUTPUT_PATH + "clean.wav", deg=OUTPUT_PATH + "output.wav", sr=22050)

    if pesq is None:
        print("Error : ", sample_name)
        return None, None

    clean_waveform = librosa_waveform(curr_clean, "Clean")
    noisy_waveform = librosa_waveform(curr_origin, "Noisy")
    output_waveform = librosa_waveform(curr_output, "Output")

    if glob_step == 0:
        tb.add_figure(f"{sample_name} Clean Waveform", clean_waveform, global_step=glob_step)
        tb.add_figure(f"{sample_name} Noisy Waveform", noisy_waveform, global_step=glob_step)
    tb.add_figure(f"{sample_name} Output Waveform", output_waveform, global_step=glob_step)

    clean_spec = librosa_spectrogram(curr_clean, "Clean")
    noisy_spec = librosa_spectrogram(curr_origin, "Noisy")
    output_spec = librosa_spectrogram(curr_output, "Output")

    if glob_step == 0:
        tb.add_figure(f"{sample_name} Clean Spectrogram", clean_spec, global_step=glob_step)
        tb.add_figure(f"{sample_name} Noisy Spectrogram", noisy_spec, global_step=glob_step)
    tb.add_figure(f"{sample_name} Output Spectrogram", output_spec, global_step=glob_step)

    plt.close('all')

    tb.add_scalar(f"{sample_name} PESQ", pesq, global_step=glob_step)
    tb.add_scalar(f"{sample_name} SSNR", segSNR, global_step=glob_step)

    return pesq, segSNR


if __name__ == "__main__":

    # ########################## !! MODIFY !! ########################### #

    DEFAULT_TB_LOG = "./runs/metrics_SPL_final"
    models = get_ckpts("./wave_u_net_checkpoints_SPL/final/")
    sampling_num = 100
    testset_list = "testset/testset_sort_pesq.txt"

    # ################################################################### #
    import os

    if not os.path.isdir(DEFAULT_TB_LOG):
        os.mkdir(DEFAULT_TB_LOG)

    sample_list = dataset_list = [line.rstrip('\n') for line in open(testset_list, "r")]
    from random import sample

    if sampling_num is not None:
        print(sample_list[:sampling_num])
        sample_list = sample_list[:sampling_num]

    # sample_audio_list = ["p232_023.wav", "p232_020.wav", "p257_056.wav", "p257_256.wav", "p257_334.wav", "p232_244.wav"]

    model = Model()
    models = sum(models, [])


    pesq, segSNR = None, None
    tb = SummaryWriter(DEFAULT_TB_LOG)
    model_step = 0

    for modelckpt in tqdm(models):
        pesq_list = []
        segSNR_list = []
        for sample in tqdm(sample_list):
            sample_name = sample[-12:]
            pesq, segSNR = sample_audio_evaluation_tensorboard(sample_name, DEFAULT_TB_LOG, modelckpt, model_step)
            if pesq is None:
                continue

            if pesq is not None and segSNR is not None:
                pesq_list.append(pesq)
                segSNR_list.append(segSNR)
        print("PESQ : ", np.mean(np.array(pesq_list)), "SSNR :", np.mean(np.array(segSNR_list)), "\n" )
        tb.add_scalar(f"PESQ", np.mean(np.array(pesq_list)), global_step=model_step)
        tb.add_scalar(f"SSNR", np.mean(np.array(segSNR_list)), global_step=model_step)

        model_step += 1

