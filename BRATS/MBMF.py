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

import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gym
from gym import spaces
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Simple2DUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TumorClassifier(nn.Module):
    def __init__(self, in_dim=32, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)


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

import os
import gym
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from gym import spaces
from scipy.ndimage import gaussian_filter1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_score(pred, target, epsilon=1e-6):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def smooth_curve(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma)

def advanced_plot_dice(raw_dice_scores, smoothed_dice_scores):
    plt.figure(figsize=(12, 6))
    plt.plot(raw_dice_scores, label='Raw Dice Score', color='red', linestyle='--', marker='o', alpha=0.5)
    plt.plot(smoothed_dice_scores, label='Smoothed Dice Score (Gaussian)', color='blue', linewidth=2)
    plt.fill_between(range(len(raw_dice_scores)), smoothed_dice_scores - 0.01, smoothed_dice_scores + 0.01, color='blue', alpha=0.1)
    plt.title("Segmentation Dice Score Across Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Dice Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

class TumorGrowthEnv(gym.Env):
    def __init__(self, volume_df, brats_data_dir):
        super().__init__()
        self.volume_df = volume_df
        self.brats_data_dir = brats_data_dir
        self.feature_dim = 32
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 + self.feature_dim,), dtype=np.float32)
        self.max_steps = 20
        self.segmenter = Simple2DUNet().to(device)
        self.classifier = TumorClassifier().to(device)
        self.dice_scores = []

    def _load_real_features(self, patient_id):
        patient_dir = os.path.join(self.brats_data_dir, patient_id)
        return extract_features_for_patient(patient_dir)

    def _segment_tumor(self, patient_id):
        patient_dir = os.path.join(self.brats_data_dir, patient_id)
        modalities = []
        seg_mask = None
        for file in os.listdir(patient_dir):
            path = os.path.join(patient_dir, file)
            if "seg" in file.lower() and file.endswith(".nii.gz"):
                seg_nii = nib.load(path).get_fdata()
                mid_slice = seg_nii.shape[2] // 2
                seg_mask = seg_nii[:, :, mid_slice]
            for mod in ["t1", "t1ce", "t2", "flair"]:
                if mod in file.lower() and file.endswith(".nii.gz"):
                    img = nib.load(path).get_fdata()
                    img = np.clip(img, 0, np.percentile(img, 99))
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    mid_slice = img.shape[2] // 2
                    modalities.append(img[:, :, mid_slice])
        if len(modalities) != 4 or seg_mask is None:
            return None, 0.0
        input_tensor = np.stack(modalities)
        input_tensor = torch.tensor(input_tensor).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = self.segmenter(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        dice = dice_score(pred, seg_mask)
        self.dice_scores.append(dice)
        return pred, dice

    def _classify_tumor(self, features):
        x = torch.tensor(features).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = self.classifier(x)
            prediction = torch.argmax(logits, dim=1).item()
        return prediction

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
        self.segmentation_mask, self.dice_score_seg = self._segment_tumor(self.patient_id)
        self.predicted_class = self._classify_tumor(self.tumor_features)
        self.current_size = 0.3
        self.growth_rate = np.clip(np.random.normal(0.07, 0.015), 0.03, 0.1)
        self.decay_rate = np.clip(np.random.normal(0.03, 0.01), 0.01, 0.05)
        state = np.concatenate(([self.current_size, self.delta, self.norm_volume, self.growth_rate, self.age_norm, self.sex_norm], self.tumor_features))
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
        state = np.concatenate(([self.current_size, self.delta, self.norm_volume, self.growth_rate, self.age_norm, self.sex_norm], self.tumor_features))
        return state, reward, self.done, {}

def summarize_and_plot_dice_scores(env):
    if len(env.dice_scores) > 1:
        smoothed = smooth_curve(env.dice_scores, sigma=2)
        advanced_plot_dice(env.dice_scores, smoothed)


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
    modalities = {"t1": None, "t1ce": None, "t2": None, "flair": None}
    for file in os.listdir(patient_dir):
        filepath = os.path.join(patient_dir, file)
        if file.endswith(".nii.gz"):
            fname = file.lower()
            if "t1c" in fname or "t1ce" in fname:
                modalities["t1ce"] = nib.load(filepath).get_fdata()
            elif "t1" in fname and "t1c" not in fname:
                modalities["t1"] = nib.load(filepath).get_fdata()
            elif "t2" in fname:
                modalities["t2"] = nib.load(filepath).get_fdata()
            elif "flair" in fname:
                modalities["flair"] = nib.load(filepath).get_fdata()

    if any(v is None for v in modalities.values()):
        return np.zeros(32, dtype=np.float32)

    features = []
    for mod in ["t1", "t1ce", "t2", "flair"]:
        img = modalities[mod]
        flat = img.flatten()
        features.extend([
            flat.mean(), flat.std(), skew(flat), kurtosis(flat),
            np.percentile(flat, 1), np.percentile(flat, 99),
            flat.min(), flat.max()
        ])
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
import gym
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from gym import spaces
from scipy.ndimage import gaussian_filter1d
from torch import nn, optim
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DynamicsNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim)
        )
    def forward(self, x):
        return self.net(x)

def dice_score(pred, target, epsilon=1e-6):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    inter = np.sum(pred * target); union = np.sum(pred) + np.sum(target)
    return (2*inter + epsilon) / (union + epsilon)

def smooth_curve(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma)

def advanced_plot_dice(raw, smooth):
    plt.figure(figsize=(10,5))
    plt.plot(raw, '--o', alpha=0.5, label='Raw Dice')
    plt.plot(smooth, '-k', label='Smoothed Dice')
    plt.fill_between(range(len(raw)), smooth-0.01, smooth+0.01, alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel("Dice Score"); plt.legend();
    plt.title("Segmentation Dice Score over Episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metrics(metrics_dict):
    titles = list(metrics_dict.keys())
    plt.figure(figsize=(15, 8))
    for i, title in enumerate(titles):
        plt.subplot(2, (len(titles) + 1) // 2, i + 1)
        plt.plot(metrics_dict[title])
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(title)
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_mbmf_dqn(env, episodes=500, batch_size=64, gamma=0.99, lr=1e-3,
                   dyn_lr=1e-3, dyn_weight=0.1, tau=0.005):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = DQN(obs_dim, act_dim).to(device)
    target = DQN(obs_dim, act_dim).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    dynamics = DynamicsNet(obs_dim, act_dim).to(device)

    opt_q = optim.Adam(policy.parameters(), lr=lr)
    opt_dyn = optim.Adam(dynamics.parameters(), lr=dyn_lr)
    replay = PrioritizedReplayBuffer(alpha=0.6)

    dice_scores = []
    rewards_all, all_losses = [], []
    dyn_losses = []
    q_values = []
    td_errors = []
    episode_lengths = []
    reward_loss_ratios = []
    gradient_norms = []
    f1_scores = []
    accuracies = []

    for ep in range(episodes):
        state = env.reset()
        total_r = 0
        done = False
        loss = None
        loss_dyn = None
        steps = 0
        q_vals = []
        td_errs = []
        f1 = getattr(env, "f1_score", None)
        acc = getattr(env, "accuracy", None)
        grad_norm = 0.0

        if hasattr(env, "dice_score_seg"):
            dice_scores.append(env.dice_score_seg)

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_pred = policy(s_t)
                action = q_pred.argmax(dim=1).item()
                q_vals.append(q_pred.max().item())

            nxt, reward, done, _ = env.step(action)
            replay.push(state, action, reward, nxt, done)
            state = nxt
            total_r += reward
            steps += 1

            if len(replay.buffer) < batch_size:
                continue

            s, a, r, s1, d, weights, ids = replay.sample(batch_size, beta=min(1, 0.4 + ep / episodes))
            a_oh = nn.functional.one_hot(a, act_dim).float()
            s, a, r, s1, d, w = [x.to(device) for x in (s, a, r, s1, d, weights)]

            q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_a = policy(s1).argmax(dim=1, keepdim=True)
                q_tgt = target(s1).gather(1, next_a).squeeze(1)
                y = r + gamma * q_tgt * (1 - d)

            td = (q - y).abs().detach().cpu().numpy()
            td_errs.extend(td.tolist())
            replay.update_priorities(ids, td + 1e-6)

            loss_q = (nn.SmoothL1Loss(reduction='none')(q, y) * w).mean()

            dyn_in = torch.cat([s, a_oh], dim=1)
            s1_pred = dynamics(dyn_in)
            loss_dyn = nn.MSELoss()(s1_pred, s1)
            loss = loss_q + dyn_weight * loss_dyn

            opt_q.zero_grad()
            opt_dyn.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), 10).item()
            opt_q.step()
            opt_dyn.step()

            for tp, p in zip(target.parameters(), policy.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        rewards_all.append(total_r)
        all_losses.append(loss.item() if loss is not None else 0.0)
        dyn_losses.append(loss_dyn.item() if loss_dyn is not None else 0.0)
        q_values.append(np.mean(q_vals) if q_vals else 0.0)
        td_errors.append(np.mean(td_errs) if td_errs else 0.0)
        episode_lengths.append(steps)
        reward_loss_ratios.append(total_r / (loss.item() + 1e-6) if loss is not None else 0.0)
        gradient_norms.append(grad_norm)
        if f1: f1_scores.append(f1)
        if acc: accuracies.append(acc)

    if dice_scores:
        adv = smooth_curve(dice_scores, sigma=2)
        advanced_plot_dice(dice_scores, adv)

    metrics = {
        "Total Rewards": rewards_all,
        "Loss Curve": all_losses,
        "Dynamics Loss": dyn_losses,
        "Mean Q-values": q_values,
        "TD Error": td_errors,
        "Episode Lengths": episode_lengths,
        "Reward/Loss Ratio": reward_loss_ratios,
        "Gradient Norms": gradient_norms,
    }

    if f1_scores:
        metrics["F1 Score"] = f1_scores
    if accuracies:
        metrics["Accuracy"] = accuracies

    plot_metrics(metrics)

    return policy, dynamics


if __name__ == "__main__":
    root_data_dir = "C:/Users/rohan/OneDrive/Desktop/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth-003/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
    validation_dir = "C:/Users/rohan/OneDrive/Desktop/MICCAI2024-BraTS-GoAT-ValidationData-002/MICCAI2024-BraTS-GoAT-ValidationData"
    compute_tumor_volumes(root_data_dir)
    volume_df = pd.read_csv("tumor_volumes.csv")
    env = TumorGrowthEnv(volume_df, root_data_dir)
    trained_policy, trained_dynamics = train_mbmf_dqn(env)
    val_reward = evaluate_on_validation(
        trained_policy,
        validation_dir,
        TumorGrowthEnv,
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Validation Mean Reward: {val_reward:.4f}")
