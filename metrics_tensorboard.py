from metrics import *
import glob

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



def sample_audio_evaluation_tensorboard(sample_audio, tb_log_dir, model_path, glob_step):
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
    if len(length) == 0:
        return None, None
    curr_clean = clean_data[:length]
    curr_origin = origin_data[:length]

    maxv = np.iinfo(np.int16).max
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

    #
    DEFAULT_TB_LOG = "./runs/metrics"
    sample_audio = "p232_023.wav"
    CKPTS_PATH = ""
    #

    models = get_ckpts("./wave_u_net_checkpoints/")
    model = Model()
    models = sum(models,[])
    sample_list = glob.glob("./testset/clean/*.wav")
    pesq_list = []
    segSNR_list = []

    pesq, segSNR = None, None
    for sample in tqdm(sample_list):
        glob_step = 0
        for modelckpt in tqdm(models):
            sample_name = sample[-12:]
            pesq, segSNR = sample_audio_evaluation_tensorboard(sample_name, DEFAULT_TB_LOG, modelckpt, glob_step)
            glob_step += 1

        if pesq is not None and segSNR is not None:
            pesq_list.append(pesq)
            segSNR_list.append(segSNR)

    tb = SummaryWriter(DEFAULT_TB_LOG)
    tb.add_scalar(f"PESQ", np.mean(pesq_list), global_step=0)
    tb.add_scalar(f"SSNR", np.mean(segSNR_list), global_step=0)


