import os

sample_rate = 16000
n_frames = 64
frame_period = 5.0
coded_dim = 128
device = 'cuda'
load_epoch = 0
load_model_path = os.path.join(os.sep, "home", "results", "debug", "ckpts")
gpu_ids = [0]
wav_max_size = sample_rate * 60 * 5
preprocess_max_batch = 100

voice_speaker = ["lewis", "kevin"]
voice_files = list()
voice_preprocess_dir = list()

for p in voice_speaker:
    voice_files.append(os.path.join(".", "dataset", p, "*.wav"))
    voice_preprocess_dir.append(os.path.join(".", "preprocess_data", p))

pre_conver_path = os.path.join(".", "pre_conver")
pre_conver_types = ["m4a", "mp3", "mp4"]
conver_save_path = os.path.join(".", "dataset")

test_wav_save = os.path.join(".", "test")
