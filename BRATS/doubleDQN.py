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


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


def train_double_dqn(env, episodes=500, batch_size=64, gamma=0.99, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    epsilon_start, epsilon_final, epsilon_decay = 1.0, 0.01, 300
    steps_done = 0
    rewards_all = []

    for episode in tqdm(range(episodes), desc="Training DQN"):
        state = env.reset()
        total_reward, done = 0, False

        while not done:
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1

            if random.random() > epsilon:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy_net(state_tensor).max(1)[1].item()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) > batch_size:
                s, a, r, s1, d = replay_buffer.sample(batch_size)
                s, a, r, s1, d = s.to(device), a.to(device), r.to(device), s1.to(device), d.to(device)

                q_vals = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                next_q_vals = target_net(s1).max(1)[0]
                expected = r + gamma * next_q_vals * (1 - d)

                loss = nn.MSELoss()(q_vals, expected.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % 50 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        rewards_all.append(total_reward)

    plt.plot(rewards_all)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.grid(True)
    plt.show()
    return policy_net


if __name__ == "__main__":
    root_data_dir = "C:/Users/rohan/OneDrive/Desktop/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth-003/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
    compute_tumor_volumes(root_data_dir)
    volume_df = pd.read_csv("tumor_volumes.csv")
    env = TumorGrowthEnv(volume_df, root_data_dir)
    trained_policy = train_double_dqn(env)
