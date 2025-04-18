import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli

import numpy as np
from collections import deque
import random

from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
import gymnasium as gym


# Batched AC
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, num_buttons, hidden_dim=64):
        super().__init__()
        # critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_buttons),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, obs_dim)
        x = x.to(torch.float32)
        x = x.reshape(x.size(0), -1)
        logits = self.actor(x)          # (batch, num_buttons)
        values = self.critic(x).squeeze(-1)  #  (batch,)
        return logits, values

    def get_actions(self, obs: torch.Tensor):
        logits, values = self.forward(obs)
        dist = Bernoulli(logits=logits)
        actions = dist.sample()              # (batch,)
        log_probs = dist.log_prob(actions).sum(dim=1)   # (batch,)
        return actions, log_probs, values


# PPOAgent that does exactly one vectorized rollout + update
class PPOAgent:
    def __init__(self, obs_dim, num_buttons,
                 lr=3e-4, gamma=0.99,
                 clip_eps=0.2, epochs=10,
                 n_envs=8, rollout_length=256,
                 device='cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.n_envs = n_envs
        self.T = rollout_length

        self.net = ActorCritic(obs_dim, num_buttons).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def rollout(self, envs):
        """
        Run T steps in parallel on n_envs, return
        flattened batches of obs, act, logp, val, rew, done
        each of shape (T*n_envs, ...)
        """
        obs_batch, act_batch, logp_batch, val_batch = [], [], [], []
        rew_batch, done_batch = [], []

        obs = envs.reset()                     
        for _ in range(self.T):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            actions, logps, values = self.net.get_actions(obs_t)
            actions_np = actions.cpu().numpy()
            step_out = envs.step(actions_np)
            if len(step_out) == 5:
            
            	nxt_obs, rews, terms, truncs, infos = step_out
            	dones = np.logical_or(terms, truncs)
            
            else:
            	nxt_obs, rews, dones, infos = step_out

            # collect
            obs_batch.append(obs.copy())
            act_batch.append(actions.cpu().numpy())
            logp_batch.append(logps.detach().cpu().numpy())
            val_batch.append(values.detach().cpu().numpy())
            rew_batch.append(rews)
            done_batch.append(dones)

            obs = nxt_obs

        # stack → (T, n_envs, …) then flatten to (T*n_envs, …)
        def _flatten(x): 
            return np.stack(x, axis=0).reshape(-1, *x[0].shape[1:])
        obs_f    = torch.as_tensor(_flatten(obs_batch), dtype=torch.float32, device=self.device)
        acts_f   = torch.as_tensor(_flatten(act_batch),   device=self.device)
        logp_f   = torch.as_tensor(_flatten(logp_batch),  device=self.device)
        vals_f   = torch.as_tensor(_flatten(val_batch),   device=self.device)
        rews_f   = _flatten(rew_batch)    # as numpy for return comp
        done_f   = _flatten(done_batch)

        # compute discounted returns & advantages:
        returns = []
        G = np.zeros(self.n_envs, dtype=np.float32)
        for r, d in zip(rew_batch[::-1], done_batch[::-1]):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G.copy())
        returns = np.stack(returns, axis=0).reshape(-1)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        advs = returns - vals_f
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return obs_f, acts_f, logp_f, returns, advs

    def update(self, obs, acts, old_logp, returns, advs):
        for _ in range(self.epochs):
            logits, values = self.net(obs)                     
            dist    = Bernoulli(logits=logits)                
            logp    = dist.log_prob(acts).sum(dim=1)          
            ratio   = torch.exp(logp - old_logp)              
            
            unclipped   = ratio * advs
            clipped     = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advs
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            value_loss  = F.mse_loss(values, returns)

            loss = policy_loss + 0.5 * value_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


    def train(self, envs, total_updates):
        for u in range(total_updates):
            obs, acts, logp, ret, adv = self.rollout(envs)
            self.update(obs, acts, logp, ret, adv)
            print(f"Update {u+1}/{total_updates} complete")
            # if (u+1) % 10 == 0:
            #    print(f"Update {u+1}/{total_updates} complete")

def save_weights(agent: PPOAgent, path: str):
    
    torch.save(agent.net.state_dict(), path)
    print(f"Saved model weights to {path}")

def load_weights(agent: PPOAgent, path: str, device: torch.device):
   
    state = torch.load(path, map_location=device)
    agent.net.load_state_dict(state)
    agent.net.to(device)
    agent.net.eval()
    print(f"Loaded model weights from {path}")

def test_and_display(agent,game_ids,weights_path,device):
    load_weights(agent, weights_path, device)
    import time
    for game_id in game_ids:
    
        print(f"\n=== Testing {game_id} ===")
        # make sure the env supports human rendering
        env = gym.make(game_id, render_mode="human")
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # render
            env.render()
            # pick action
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = agent.net(obs_t)
            action = torch.argmax(logits, dim=1).item()

            # step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
	    # slow down
            time.sleep(0.05)   
            
        print(f"Episode finished, total reward: {total_reward:.2f}")
        env.close()

def make_env():
    
    return gym.make("CartPole-v1")   
    
if __name__ == "__main__":
    # training setup
    n_envs          = 8
    rollout_length  = 256
    total_updates   = 100
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # vectorized training env
    envs = DummyVecEnv([make_env for _ in range(n_envs)])
    obs_dim = envs.observation_space.shape[0]
    num_buttons = int(np.prod(envs.observation_space.shape))

    agent = PPOAgent(
        obs_dim, num_buttons,
        lr=3e-4, gamma=0.99,
        clip_eps=0.2, epochs=4,
        n_envs=n_envs,
        rollout_length=rollout_length,
        device=device
    )

    # train
    agent.train(envs, total_updates)

    # save trained weights
    weights_file = "ppo_actor_critic.pth"
    save_weights(agent, weights_file)

    # now test on eight distinct games:
    games = [
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "LunarLander-v2",
        "Pendulum-v1",
        "MountainCarContinuous-v0",
        "CartPole-v0",
        "Acrobot-v1",
    ]
    test_and_display(agent, games, weights_file, device)

