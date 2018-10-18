#!/usr/bin/env python
import os
import librosa  # pip install librosa
from tqdm import tqdm # 显示进度条
import numpy as np
import scipy.io.wavfile as wav

# FLAGS.n_inputs  40
# 获取文件路径
def get_wav_files(parent_dir,  sub_dirs):
    wav_files = []
    for l,  sub_dir in enumerate(sub_dirs):
        wav_path = os.sep.join([parent_dir,  sub_dir])
        for (dirpath,  dirnames,  filenames) in os.walk(wav_path):
            for filename in filenames:
                filename_path = os.sep.join([dirpath,  filename])
                wav_files.append(filename_path)#filename_path:audio\5\3xing\xing020.wav
    return wav_files

# 获取文件mfcc特征
def extract_features(wav_files):
    inputs = []
    # for wav_file in tqdm(wav_files):
    # for wav_file in wav_files:
    audio, fs = librosa.load(wav_files)
    # audio = audio[0:int(3.5 * fs)]
    # 获取音频mfcc特征[n_steps,  n_inputs](分帧的数量，特征)
    mfccs = np.transpose(librosa.feature.mfcc(y=audio,  sr=fs,  n_mfcc=40),  [1, 0])
    inputs.append(mfccs.tolist())
    return inputs

#训练时获取文件fbank特征
# def extract_fbank_features(wav_files):
#     inputs = []
#     for wav_file in tqdm(wav_files):
#     # for wav_file in wav_files:
#         # 读入音频文件
#         (rate, sig) = wav.read(wav_file)
#         fbank_feat = base.logfbank(sig, rate)
#         inputs.append(fbank_feat.tolist())
#     return inputs
def pre_fbank_features(wav_files):
    inputs = []
    #载入数据
    sample_rate, signal = wav.read(wav_files)
    #预加重y(t)=x(t)−αx(t−1)
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    #分帧，25毫秒的帧大小，frame_size = 0.025和10毫秒的步幅（15毫秒重叠），frame_stride = 0.01。
    frame_size = 0.05
    frame_stride = 0.02
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))#补0操作
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile  (np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

    #加窗
    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # 具体实现
    #傅里叶变换
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  #功率谱
    #转换成Mel频率
    low_freq_mel = 0
    nfilt = 26
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  #  Hz 转换成 Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Mel比例尺等间距
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  #  Mel 转换成 Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # 左
        f_m = int(bin[m])  # 中
        f_m_plus = int(bin[m + 1])  # 右
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    inputs.append(filter_banks.tolist())
    return inputs

def extract_fbank_features(wav_files):
    inputs = []
    for wav_file in tqdm(wav_files):
        #载入数据
        sample_rate, signal = wav.read(wav_file)
        # signal = signal[0:int(3.5 * sample_rate)]
        #
        # signal, sample_rate = librosa.load(wav_file)
        # sample_rate = 16000
        # wav_file = wave.open(wav_file)
        # sample_rate, nframes = wav_file.getparams()[2:4]
        # frm_data = wav_file.readframes(nframes)
        # signal = np.fromstring(frm_data, dtype=np.short)

        #预加重y(t)=x(t)−αx(t−1)
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        # fig = plt.figure()
        # ax1 = fig.add_subplot(2, 2, 1)
        # ax1.plot(signal)
        # ax2 = fig.add_subplot(2, 2, 2)
        # ax2.plot(emphasized_signal)
        # plt.show()
        #分帧，25毫秒的帧大小，frame_size = 0.025和10毫秒的步幅（15毫秒重叠），frame_stride = 0.01。
        frame_size = 0.05
        frame_stride = 0.02
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))#补0操作
        pad_signal = np.append(emphasized_signal, z)
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile  (np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

        frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

        # ax3 = fig.add_subplot(2, 2, 3)
        # ax3.plot(frames)
        #加窗
        frames *= np.hamming(frame_length)
        # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # 具体实现
        #傅里叶变换
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  #功率谱
        #转换成Mel频率
        low_freq_mel = 0
        nfilt = 26
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  #  Hz 转换成 Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Mel比例尺等间距
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  #  Mel 转换成 Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # 左
            f_m = int(bin[m])  # 中
            f_m_plus = int(bin[m + 1])  # 右
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

        # num_ceps = 40
        # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
        # (nframes, ncoeff) = mfcc.shape
        # n = np.arange(ncoeff)
        # cep_lifter = 22
        # lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        # mfcc *= lift
        # mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        # print(mfcc)
        inputs.append(filter_banks.tolist())
    return inputs


#获取对应label
def extract_labels(wav_files):
    labels_set = []
    labels_repeat = []
    labels_unique = []
    for wav_file in wav_files:
        label_temp = wav_file.split('\\')
        label = label_temp[2]
        labels_repeat.append(label)
    for x in labels_repeat:
        if x not in labels_unique:
            labels_unique.append(x)
    for y in labels_repeat:
        y_index = labels_unique.index(y)
        labels_set.append(y_index)
    print(labels_unique)
    return np.array(labels_set,  dtype=np.int)

if __name__ == '__main__':
    wav_files = get_wav_files("../audio-8k-cm", "1,2,3,4,5")
    # train_8k_mfcc_features = extract_features(wav_files)
    # np.save('../features_npy/train_8k_mfcc_features.npy', train_8k_mfcc_features)

    # train_me_fbank_features = extract_fbank_features(wav_files)
    # np.save('train_me4_fbank_features.npy', train_me_fbank_features)
    train_labels = extract_labels(wav_files)
    # np.save('train_8k_labels.npy', train_labels)

