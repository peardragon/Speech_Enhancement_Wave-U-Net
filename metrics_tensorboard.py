from metrics import *


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
        curr_output, length = speech_enhancement(model, origin_data)
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

        # ref : clean one  / deg : predict one
        pesq, segSNR, _ = Eval(ref=OUTPUT_PATH + "clean.wav", deg=OUTPUT_PATH + "output.wav", sr=22050)

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

