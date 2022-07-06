import os
import mel
import config
import numpy as np
import pickle
import glob


def pack_up(values):
    pack_value = list()
    overflow = list()

    for value in values:
        for index in range(0, value.shape[-1], config.n_frames):
            start, end = index, index + config.n_frames
            slice_value = value[:, start:end]

            if slice_value.shape[-1] == config.n_frames:
                pack_value.append(slice_value)
            else:
                overflow.append(slice_value)

    # overflow dataset join to build new dataset
    overflow = np.concatenate([*overflow], axis=1)

    for index in range(0, overflow.shape[-1], config.n_frames):
        start, end = index, index + config.n_frames
        slice_value = overflow[:, start:end]

        if slice_value.shape[-1] == config.n_frames:
            pack_value.append(slice_value)

    return pack_value


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def batch_world_encode():
    for spk, files, save_dir_path in zip(config.voice_speaker, config.voice_files, config.voice_preprocess_dir):
        wavs = mel.load_wavs(files, config.sample_rate)

        for batch in range(0, len(wavs), config.preprocess_max_batch):
            start, end = batch, batch + config.preprocess_max_batch
            batch_of_wavs = wavs[start:end]

            f0s, timeaxes, sps, aps, coded_sps = mel.world_encode_data(batch_of_wavs, config.sample_rate, config.frame_period, config.coded_dim)

            file_path = os.path.join(save_dir_path, f"{spk}_{str(batch).zfill(5)}_")
            save_pickle(f0s, file_path + "f0s.pickle")
            save_pickle(timeaxes, file_path + "timeaxes.pickle")
            save_pickle(sps, file_path + "sps.pickle")
            save_pickle(aps, file_path + "aps.pickle")
            save_pickle(coded_sps, file_path + "coded_sps.pickle")


def get_mean_std():
    for spk, save_dir_path in zip(config.voice_speaker, config.voice_preprocess_dir):
        f0s_paths = glob.glob(os.path.join(save_dir_path, "*_f0s.pickle"))

        all_f0s = list()

        for path in f0s_paths:
            f0s = load_pickle_file(path)
            all_f0s = [*all_f0s, *f0s]

        mean, std = mel.logf0_statistics(f0s)

        np.save(os.path.join(save_dir_path, f"{spk}_mean.npy"), mean)
        np.save(os.path.join(save_dir_path, f"{spk}_std.npy"), std)


def get_mel():
    for spk, save_dir_path in zip(config.voice_speaker, config.voice_preprocess_dir):
        coded_sps_paths = glob.glob(os.path.join(save_dir_path, "*_coded_sps.pickle"))

        all_coded_sps = list()

        for path in coded_sps_paths:
            coded_sps = load_pickle_file(path)
            all_coded_sps = [*all_coded_sps, *coded_sps]

        coded_sps_T = mel.transpose_in_list(coded_sps)

        save_pickle(coded_sps_T, os.path.join(save_dir_path, f"{spk}_mel.pickle"))


def get_mel_pack_up():
    for spk, save_dir_path in zip(config.voice_speaker, config.voice_preprocess_dir):
        mel_batchs = load_pickle_file(os.path.join(save_dir_path, f"{spk}_mel.pickle"))

        mel_pack = pack_up(mel_batchs)
        np.save(os.path.join(save_dir_path, f"{spk}_mel_format.npy"), mel_pack)


def get_norm():
    for spk, save_dir_path in zip(config.voice_speaker, config.voice_preprocess_dir):
        mels = np.load(os.path.join(save_dir_path, f"{spk}_mel_format.npy"))

        norm_mel, norm_mean, norm_std = mel.coded_sps_normalization_fit_transoform(mels)

        np.save(os.path.join(save_dir_path, f"{spk}_norm_mel_format.npy"), norm_mel)
        np.save(os.path.join(save_dir_path, f"{spk}_norm_mean.npy"), norm_mean)
        np.save(os.path.join(save_dir_path, f"{spk}_norm_std.npy"), norm_std)


if __name__ == '__main__':
    batch_world_encode()
    get_mean_std()
    get_mel()
    get_mel_pack_up()
    get_norm()
