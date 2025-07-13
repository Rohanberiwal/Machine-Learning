import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
import random


def compute_tumor_volumes(root_dir):
    seg_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("seg.nii.gz"):
                seg_paths.append(os.path.join(root, file))

    volume_data = []
    for path in tqdm(seg_paths, desc="Computing tumor volumes"):
        img = nib.load(path)
        data = img.get_fdata()
        voxel_volume = np.prod(img.header.get_zooms())
        tumor_voxels = np.sum(data > 0)
        volume_mm3 = tumor_voxels * voxel_volume
        patient_id = os.path.basename(os.path.dirname(path))
        volume_data.append({"patient_id": patient_id, "tumor_voxels": tumor_voxels, "tumor_volume_mm3": volume_mm3})

    df = pd.DataFrame(volume_data)
    df.to_csv("tumor_volumes.csv", index=False)
    print("Tumor volume extraction complete. Saved to tumor_volumes.csv")


def extract_features_for_patient(patient_dir):
    modalities = {"t1": None, "t1ce": None, "t2": None, "flair": None, "seg": None}
    for file in os.listdir(patient_dir):
        filepath = os.path.join(patient_dir, file)
        if file.endswith(".nii.gz"):
            fname = file.lower()
            if "t1ce" in fname:
                modalities["t1ce"] = nib.load(filepath).get_fdata()
            elif "t1" in fname and "t1ce" not in fname:
                modalities["t1"] = nib.load(filepath).get_fdata()
            elif "t2" in fname:
                modalities["t2"] = nib.load(filepath).get_fdata()
            elif "flair" in fname:
                modalities["flair"] = nib.load(filepath).get_fdata()
            elif "seg" in fname:
                modalities["seg"] = nib.load(filepath).get_fdata()

    if any(v is None for v in modalities.values()):
        return np.zeros(32, dtype=np.float32)

    tumor_mask = modalities["seg"] > 0
    if np.sum(tumor_mask) == 0:
        return np.zeros(32, dtype=np.float32)

    features = []
    for mod_name in ["t1", "t1ce", "t2", "flair"]:
        img = modalities[mod_name]
        tumor_region = img[tumor_mask]
        mean_intensity = tumor_region.mean()
        std_intensity = tumor_region.std()
        skew_intensity = skew(tumor_region)
        kurt_intensity = kurtosis(tumor_region)
        tumor_region_scaled = ((tumor_region - tumor_region.min()) / (tumor_region.max() - tumor_region.min()) * 255).astype(np.uint8)
        if len(tumor_region_scaled) < 10:
            contrast = correlation = energy = homogeneity = 0.0
        else:
            glcm = graycomatrix(tumor_region_scaled.reshape(-1, 1), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        features.extend([mean_intensity, std_intensity, skew_intensity, kurt_intensity, contrast, correlation, energy, homogeneity])
    return np.array(features, dtype=np.float32)


class TumorGrowthEnv(gym.Env):
    def __init__(self, volume_df, brats_data_dir):
        super().__init__()
        self.volume_df = volume_df
        self.brats_data_dir = brats_data_dir
        self.feature_dim = 32
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 + self.feature_dim,), dtype=np.float32)
        self.max_steps = 20

    def _load_real_features(self, patient_id):
        patient_dir = os.path.join(self.brats_data_dir, patient_id)
        return extract_features_for_patient(patient_dir)

    def reset(self):
        self.steps = 0
        self.done = False
        self.delta = 0.0
        self.prev_size = 0.3

        sample = self.volume_df.sample(1).iloc[0]
        self.patient_id = sample['patient_id']
        self.tumor_volume = sample['tumor_volume_mm3']
        self.age = sample.get('age', 50)
        self.sex = sample.get('sex', 1)

        max_volume = self.volume_df['tumor_volume_mm3'].max()
        self.norm_volume = self.tumor_volume / max_volume if max_volume > 0 else 0.0

        self.age_norm = (self.age - 40) / 40
        self.sex_norm = float(self.sex)
        self.tumor_features = self._load_real_features(self.patient_id)

        self.current_size = 0.3
        self.growth_rate = np.clip(np.random.normal(0.07, 0.015), 0.03, 0.1)
        self.decay_rate = np.clip(np.random.normal(0.03, 0.01), 0.01, 0.05)

        state = np.concatenate((
            [self.current_size, self.delta, self.norm_volume, self.growth_rate, self.age_norm, self.sex_norm],
            self.tumor_features
        ))
        return state

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done")

        noise = np.random.normal(0, 0.02)
        if action == 0:
            self.current_size = min(1.0, self.current_size * (1 + self.growth_rate + noise))
            reward = -self.current_size
        elif action == 1:
            self.current_size = max(0.0, self.current_size * (1 - self.decay_rate + noise))
            reward = -self.current_size - 0.05
        else:
            self.current_size = max(0.0, self.current_size * (1 - 2 * self.decay_rate + noise))
            reward = -self.current_size - 0.15

        self.steps += 1
        self.delta = self.current_size - self.prev_size
        self.prev_size = self.current_size
        self.done = self.current_size < 0.01 or self.steps >= self.max_steps

        state = np.concatenate((
            [self.current_size, self.delta, self.norm_volume, self.growth_rate, self.age_norm, self.sex_norm],
            self.tumor_features
        ))
        return state, reward, self.done, {}



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        epsilon_in = self._f(self.in_features)
        epsilon_out = self._f(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _f(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(input, weight, bias)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, output_dim)
        )
        self.value = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        adv = self.advantage(x)
        val = self.value(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = list(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

def extract_validation_features(patient_dir):
    modalities = {"t1": None, "t1ce": None, "t2": None, "flair": None, "seg": None}
    for file in os.listdir(patient_dir):
        filepath = os.path.join(patient_dir, file)
        if file.endswith(".nii.gz"):
            fname = file.lower()
            if "t1ce" in fname:
                modalities["t1ce"] = nib.load(filepath).get_fdata()
            elif "t1" in fname and "t1ce" not in fname:
                modalities["t1"] = nib.load(filepath).get_fdata()
            elif "t2" in fname:
                modalities["t2"] = nib.load(filepath).get_fdata()
            elif "flair" in fname:
                modalities["flair"] = nib.load(filepath).get_fdata()
            elif "seg" in fname:
                modalities["seg"] = nib.load(filepath).get_fdata()

    if any(v is None for v in modalities.values()):
        return np.zeros(32, dtype=np.float32)

    tumor_mask = modalities["seg"] > 0
    if np.sum(tumor_mask) == 0:
        return np.zeros(32, dtype=np.float32)

    features = []
    for mod_name in ["t1", "t1ce", "t2", "flair"]:
        img = modalities[mod_name]
        tumor_region = img[tumor_mask]
        mean_intensity = tumor_region.mean()
        std_intensity = tumor_region.std()
        skew_intensity = skew(tumor_region)
        kurt_intensity = kurtosis(tumor_region)
        tumor_region_scaled = ((tumor_region - tumor_region.min()) / (tumor_region.max() - tumor_region.min()) * 255).astype(np.uint8)
        if len(tumor_region_scaled) < 10:
            contrast = correlation = energy = homogeneity = 0.0
        else:
            glcm = graycomatrix(tumor_region_scaled.reshape(-1, 1), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        features.extend([mean_intensity, std_intensity, skew_intensity, kurt_intensity, contrast, correlation, energy, homogeneity])
    return np.array(features, dtype=np.float32)


def evaluate_on_validation(policy_net, validation_root_dir, env_class, device):
    rewards = []
    for patient_folder in os.listdir(validation_root_dir):
        patient_dir = os.path.join(validation_root_dir, patient_folder)
        if not os.path.isdir(patient_dir): continue
        features = extract_validation_features(patient_dir)
        if np.all(features == 0): continue
        env = env_class.from_features(features)
        state = env.reset()
        total_reward, done = 0, False
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy_net(state_tensor).max(1)[1].item()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards) if rewards else 0.0

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class RAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                buffered = [None, None, None]
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                if N_sma > 5:
                    step_size = group['lr'] * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2) ** 0.5
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p.data.add_(exp_avg, alpha=-group['lr'])

class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("Base optimizer must be an instance of torch.optim.Optimizer")
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.param_groups = self.optimizer.param_groups  # <-- This is key!
        self.state = self.optimizer.state

        self.fast_weights = [
            p.clone().detach()
            for group in self.param_groups
            for p in group['params']
        ]
        for w in self.fast_weights:
            w.requires_grad = False

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.fast_weights[idx].add_(p.data - self.fast_weights[idx], alpha=self.alpha)
                    p.data.copy_(self.fast_weights[idx])
                    idx += 1

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class RAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                if N_sma > 5:
                    step_size = group['lr'] * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2) ** 0.5
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p.data.add_(exp_avg, alpha=-group['lr'])

class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("Base optimizer must be an instance of torch.optim.Optimizer")
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.fast_weights = [p.clone().detach() for group in self.param_groups for p in group['params']]
        for w in self.fast_weights:
            w.requires_grad = False
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        self.optimizer.step()
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.fast_weights[idx].add_(p.data - self.fast_weights[idx], alpha=self.alpha)
                    p.data.copy_(self.fast_weights[idx])
                    idx += 1

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def smooth(values, weight=0.9):
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def train_double_dqn(env, episodes=500, batch_size=64, gamma=0.99, lr=1e-4, target_update_tau=0.005, max_grad_norm=10, alpha=0.6, beta_start=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    base_opt = RAdam(policy_net.parameters(), lr=lr)
    optimizer = Lookahead(base_opt)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    replay_buffer = PrioritizedReplayBuffer(alpha=alpha)
    ema_model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    ema_model.load_state_dict(policy_net.state_dict())
    ema_decay = 0.995
    writer = SummaryWriter()
    all_rewards, all_losses = [], []
    all_precisions, all_recalls, all_f1s, all_accuracies = [], [], [], []
    final_preds, final_targets, final_probs, final_s = [], [], [], None
    loss_fn = FocalLoss()
    lr_tracker = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        total_reward, done = 0, False
        beta = min(1.0, beta_start + episode * (1.0 - beta_start) / episodes)
        episode_preds, episode_targets, episode_probs = [], [], []

        while not done:
            with torch.no_grad():
                s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy_net(s_tensor).max(1)[1].item()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            policy_net.reset_noise()
            target_net.reset_noise()

            if len(replay_buffer.buffer) >= batch_size:
                s, a, r, s1, d, weights, indices = replay_buffer.sample(batch_size, beta=beta)
                s, a, r, s1, d, weights = s.to(device), a.to(device), r.to(device), s1.to(device), d.to(device), weights.to(device)
                final_s = s
                with torch.no_grad():
                    next_actions = policy_net(s1).max(1)[1].unsqueeze(1)
                    next_q = target_net(s1).gather(1, next_actions).squeeze(1)
                    y = r + gamma * next_q * (1 - d)
                q_vals = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                td_errors = (q_vals - y).abs().detach().cpu().numpy()
                replay_buffer.update_priorities(indices, td_errors + 1e-6)
                logits = policy_net(s)
                q_loss = loss_fn(logits, a)
                optimizer.zero_grad()
                q_loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                lr_tracker.append(optimizer.param_groups[0]['lr'])
                for ema_p, p in zip(ema_model.parameters(), policy_net.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data * (1 - ema_decay))
                for t_p, p_p in zip(target_net.parameters(), policy_net.parameters()):
                    t_p.data.copy_(target_update_tau * p_p.data + (1.0 - target_update_tau) * t_p.data)
                all_losses.append(q_loss.item())
                writer.add_scalar("Loss", q_loss.item(), episode)
                with torch.no_grad():
                    out = policy_net(s)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    episode_preds.extend(preds)
                    episode_targets.extend(a.cpu().numpy())
                    episode_probs.extend(probs[np.arange(len(preds)), preds])

        all_rewards.append(total_reward)
        writer.add_scalar("Reward", total_reward, episode)
        if len(set(episode_targets)) > 1:
            acc = accuracy_score(episode_targets, episode_preds)
            prec = precision_score(episode_targets, episode_preds, average='macro', zero_division=0)
            rec = recall_score(episode_targets, episode_preds, average='macro', zero_division=0)
            f1 = f1_score(episode_targets, episode_preds, average='macro', zero_division=0)
        else:
            acc = prec = rec = f1 = 0.0
        all_accuracies.append(acc)
        all_precisions.append(prec)
        all_recalls.append(rec)
        all_f1s.append(f1)
        if episode == episodes - 1:
            final_preds = episode_preds
            final_targets = episode_targets
            final_probs = episode_probs

    writer.close()

    titles = ["Rewards", "Loss", "Accuracy", "Precision", "Recall", "F1 Score"]
    metrics = [all_rewards, all_losses, all_accuracies, all_precisions, all_recalls, all_f1s]
    colors = ["blue", "red", "green", "orange", "purple", "cyan"]
    plt.figure(figsize=(18, 12))
    for i, (title, data, color) in enumerate(zip(titles, metrics, colors)):
        plt.subplot(3, 2, i + 1)
        smoothed = smooth(data)
        plt.plot(smoothed, label=f"Smoothed {title}", color=color)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

    if 'td_errors' in locals():
        plt.figure(figsize=(6, 4))
        plt.hist(td_errors, bins=40, color='salmon')
        plt.title("TD Error Distribution")
        plt.xlabel("TD Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(lr_tracker, color='teal')
    plt.title("Learning Rate Schedule")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return ema_model


if __name__ == "__main__":
    root_data_dir = "C:/Users/rohan/OneDrive/Desktop/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth-003/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
    validation_dir = "C:/Users/rohan/OneDrive/Desktop/MICCAI2024-BraTS-GoAT-ValidationData-002/MICCAI2024-BraTS-GoAT-ValidationData"
    compute_tumor_volumes(root_data_dir)
    volume_df = pd.read_csv("tumor_volumes.csv")
    env = TumorGrowthEnv(volume_df, root_data_dir)
    trained_policy = train_double_dqn(env)
    val_reward = evaluate_on_validation(trained_policy, validation_dir, TumorGrowthEnv, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Validation Mean Reward: {val_reward:.4f}")
