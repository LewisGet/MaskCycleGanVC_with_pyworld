import os
import mel
import config
import numpy as np


for spk, files, save_dir_path in zip(config.voice_speaker, config.voice_files, config.voice_preprocess_dir):
    wavs = mel.load_wavs(files, config.sample_rate)
    f0s, timeaxes, sps, aps, coded_sps = mel.world_encode_data(wavs, config.sample_rate, config.frame_period, config.coded_dim)
    mean, std = mel.logf0_statistics(f0s)

    coded_sps_T = mel.transpose_in_list(coded_sps)
    norm_mel, norm_mean, norm_std = mel.coded_sps_normalization_fit_transoform(coded_sps_T)

    print('Log Pitch %s' % spk)
    print('Mean: %f, Std: %f' % (mean, std))
    print("org")
    print(coded_sps_T.shape, coded_sps.shape, mean.shape, std.shape)
    print("norm")
    print(norm_mel.shape, norm_mean.shape, norm_std.shape)

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    np.savez(os.path.join(save_dir_path, spk + "_org.npz"), mel=coded_sps_T, mean=mean, std=std)
    np.savez(os.path.join(save_dir_path, spk + '_normalization.npz'), mel=norm_mel, mean=norm_mean, std=norm_std)
