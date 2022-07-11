import os
import torch

name = "debug"

sample_rate = 16000
n_frames = 64
frame_period = 5.0
coded_dim = 128
packup_stride = 3
mask_len = 25
torch.cuda.set_device(0)
device = 'cuda'
load_epoch = 0
save_dir = os.path.join(os.sep, "home", "results")
load_model_path = os.path.join(os.sep, "home", "results", name, "ckpts")
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

start_epoch = 0
num_epochs = 10
logger_step_print = 100
epochs_save = 1

g_lr = 2e-4
d_lr = 1e-4

batch_size = 1
