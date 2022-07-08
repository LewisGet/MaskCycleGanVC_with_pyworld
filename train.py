import os
import re
import pickle
import numpy as np
from pathlib import Path

import torch
import torch.utils.data as data
import torchaudio
from torch.autograd import Variable

import time

import config
from model import Generator, Discriminator
from dataset import MelDataset
from train_logger import TrainLogger


class Training:
    def __init__(self):
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.input_shape = (config.coded_dim, config.n_frames)

        self.g_a2b = Generator(input_shape=self.input_shape).to(config.device)
        self.g_b2a = Generator(input_shape=self.input_shape).to(config.device)

        self.d_a = Discriminator(input_shape=self.input_shape).to(config.device)
        self.d_b = Discriminator(input_shape=self.input_shape).to(config.device)

        a2b_params = self.g_a2b.parameters()
        b2a_params = self.g_b2a.parameters()

        d_a_params = self.d_a.parameters()
        d_b_params = self.d_b.parameters()

        self.g_a2b_optimizer = torch.optim.Adam(a2b_params, lr=self.g_lr, betas=(0.5, 0.999))
        self.g_b2a_optimizer = torch.optim.Adam(b2a_params, lr=self.g_lr, betas=(0.5, 0.999))

        self.d_a_optimizer = torch.optim.Adam(d_a_params, lr=self.d_lr, betas=(0.5, 0.999))
        self.d_b_optimizer = torch.optim.Adam(d_b_params, lr=self.d_lr, betas=(0.5, 0.999))

        self.dataset_a = np.load(os.path.join(config.voice_preprocess_dir[0], f"{voice_speaker[0]}_norm_mel_format.npy"))
        self.dataset_b = np.load(os.path.join(config.voice_preprocess_dir[1], f"{voice_speaker[1]}_norm_mel_format.npy"))

        self.dataset = MelDataset(datasetA=self.dataset_a, datasetB=self.dataset_b, n_frames=config.n_frames, max_mask_len=config.mask_len)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

        if config.load_epoch is not 0:
            self.load_model(self.g_a2b, "g_a2b", None,self.g_a2b_optimizer)
            self.load_model(self.g_b2a, "g_b2a", None, self.g_b2a_optimizer)
            self.load_model(self.d_a, "d_a", None, self.d_a_optimizer)
            self.load_model(self.d_b, "d_b", None, self.d_b_optimizer)

        self.logger = TrainLogger(len(self.train_dataloader.dataset))

        print("dataset len")
        print(f"a : {len(self.dataset_a)}")
        print(f"a : {len(self.dataset_b)}")

    def reset_grad(self):
        self.g_a2b_optimizer.zero_grad()
        self.g_b2a_optimizer.zero_grad()

        self.d_a_optimizer.zero_grad()
        self.d_b_optimizer.zero_grad()

    def g_train(self, real_a, mask_a, real_b, mask_b):
        self.g_a2b.train()
        self.g_b2a.train()

        # real -> fake
        fake_b = self.g_a2b(real_a, mask_a)
        fake_a = self.g_b2a(real_b, mask_b)

        # org -> fake -> org_fake
        fake_b2 = self.g_a2b(fake_a, torch.ones_like(fake_a))
        fake_a2 = self.g_b2a(fake_b, torch.ones_like(fake_b))

        # org -> org
        same_b = self.g_a2b(real_b, torch.ones_like(real_b))
        same_a = self.g_b2a(real_a, torch.ones_like(real_a))

        d_fake_a = self.d_a(fake_a)
        d_fake_b = self.d_b(fake_b)

        d_fake_a2 = self.d_a(fake_a2)
        d_fake_b2 = self.d_b(fake_b2)

        d_same_a = self.d_a(same_a)
        d_same_b = self.d_b(same_b)

        g_loss_a2b = torch.mean((1 - d_fake_b) ** 2)
        g_loss_b2a = torch.mean((1 - d_fake_a) ** 2)

        g_loss_a2 = torch.mean((1 - d_fake_a2) ** 2)
        g_loss_b2 = torch.mean((1 - d_fake_b2) ** 2)

        g_loss_same_with_d_a = torch.mean((1 - d_same_a) ** 2)
        g_loss_same_with_d_b = torch.mean((1 - d_same_b) ** 2)

        g_loss_same_a = torch.mean(torch.abs(real_a - same_a))
        g_loss_same_b = torch.mean(torch.abs(real_b - same_b))

        g_total_a2b_loss = g_loss_a2b + g_loss_b2 + g_loss_same_b + g_loss_same_with_d_b
        g_total_b2a_loss = g_loss_b2a + g_loss_a2 + g_loss_same_a + g_loss_same_with_d_a

        g_total_loss = g_total_a2b_loss + g_total_b2a_loss

        self.reset_grad()
        g_total_a2b_loss.backward()
        g_total_b2a_loss.backward()
        self.g_a2b_optimizer.step()
        self.g_b2a_optimizer.step()

        return g_total_loss, g_total_a2b_loss, g_total_b2a_loss, g_loss_a2b, g_loss_b2a

    def d_train(self, real_a, mask_a, real_b, mask_b):
        self.d_a.train()
        self.d_b.train()

        fake_a = self.g_b2a(real_a, mask_a)
        fake_b = self.g_a2b(real_b, mask_b)

        fake_a2 = self.g_b2a(fake_b, torch.ones_like(fake_b))
        fake_b2 = self.g_a2b(fake_a, torch.ones_like(fake_a))

        d_real_a = self.d_a(real_a)
        d_real_b = self.d_b(real_b)

        d_fake_a = self.d_a(fake_a)
        d_fake_b = self.d_b(fake_b)

        d_fake_a2 = self.d_a(fake_a2)
        d_fake_b2 = self.d_b(fake_b2)

        d_loss_real_a = torch.mean((1 - d_real_a) ** 2)
        d_loss_real_b = torch.mean((1 - d_real_b) ** 2)

        d_loss_fake_a = torch.mean((0 - d_fake_a) ** 2)
        d_loss_fake_b = torch.mean((0 - d_fake_b) ** 2)

        d_loss_fake_a2 = torch.mean((0 - d_fake_a2) ** 2)
        d_loss_fake_b2 = torch.mean((0 - d_fake_b2) ** 2)

        d_total_loss_a = d_loss_real_a + d_loss_fake_a + d_loss_fake_a2
        d_total_loss_b = d_loss_real_b + d_loss_fake_b + d_loss_fake_b2

        d_total_loss = d_total_loss_a + d_total_loss_b

        self.reset_grad()
        d_total_loss_a.backward()
        d_total_loss_b.backward()
        self.d_a_optimizer.step()
        self.d_b_optimizer.step()

        return d_total_loss, d_total_loss_a, d_total_loss_b

    def train(self):
        for epoch in range(config.num_epochs):
            self.logger.start_epoch()

            for i, (real_a, mask_a, real_b, mask_b) in enumerate(self.train_dataloader):
                self.logger.start_iter()

                real_a, mask_a, real_b, mask_b = real_a.to(config.device), mask_a.to(config.device), real_b.to(config.device), mask_b.to(config.device)

                g_total_loss, g_total_a2b_loss, g_total_b2a_loss, g_loss_a2b, g_loss_b2a = self.g_train(real_a, mask_a, real_b, mask_b)

                d_total_loss, d_total_loss_a, d_total_loss_b = self.d_train(real_a, mask_a, real_b, mask_b)

                g_variables = re.compile("g_*loss*")
                d_variables = re.compile("d_*loss*")
                all_loss_keys = list(filter(g_variables.match, locals())) + list(filter(d_variables.match, locals()))

                loss_dict = dict()

                for loss_key in all_loss_keys:
                    loss_dict[loss_key] = locals()[loss_key].item()

                self.logger.log_iter(loss_dict=loss_dict)
                self.logger.end_iter()

                if self.logger.epoch % config.epochs_save == 0:
                    self.save(self.logger.epoch, self.g_a2b, self.g_a2b_optimizer, None, config.device, "g_a2b")
                    self.save(self.logger.epoch, self.g_b2a, self.g_b2a_optimizer, None, config.device, "g_b2a")
                    self.save(self.logger.epoch, self.d_a, self.d_a_optimizer, None, config.device, "d_a")
                    self.save(self.logger.epoch, self.d_b, self.d_b_optimizer, None, config.device, "d_b")

            self.logger.end_epoch()

    def save(self, epoch, model, optimizer, lr_scheduler, model_name):
        try:
            model_class = model.module.__class__.__name__
            model_state = model.to('cpu').module.state_dict()
            print("Saving unwrapped DataParallel module.")
        except AttributeError:
            model_class = model.__class__.__name__
            model_state = model.to('cpu').state_dict()

        ckpt_dict = {
            'ckpt_info': {'epoch': epoch},
            'model_class': model_class,
            'model_state': model_state,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        }

        model.to(config.device)

        file_name = f'{str(epoch).zfill(5)}_{model_name}.pth.tar'

        ckpt_path = os.path.join(self.ckpt_dir, file_name)
        torch.save(ckpt_dict, ckpt_path)
        print(f"Saved model to {ckpt_path}")

        # Remove a checkpoint if more than max_ckpts ckpts saved
        if self.max_ckpts:
            self.ckpt_names.append(ckpt_path)
            if len(self.ckpt_names) > self.max_ckpts:
                oldest_ckpt = os.path.join(
                    self.ckpt_dir, self.ckpt_names.pop(0))
                os.remove(oldest_ckpt)
                print(
                    f"Exceeded max number of checkpoints so deleting {oldest_ckpt}")

    def load_model(self, model, model_name=None, ckpt_path=None, optimizer=None, scheduler=None):
        ckpt_paths = sorted([name for name in os.listdir(config.load_model_path) if name.split(".", 1)[1] == "pth.tar"])

        if ckpt_path is None:
            if model_name and hasattr(self.args, 'load_epoch'):
                file_name = f'{str(config.load_epoch).zfill(5)}_{model_name}.pth.tar'
                ckpt_path = os.path.join(config.load_model_path, file_name)
            else:
                print("No checkpoint found. Failed to load load model checkpoint.")
                return

        checkpoint = torch.load(ckpt_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print(f"Loaded {checkpoint['model_class']} from {ckpt_path}")


if __name__ == "__main__":
    cycleGAN = Training()
    cycleGAN.train()
