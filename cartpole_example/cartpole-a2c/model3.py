import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class Actor(nn.Module):
    '''
    演员Actor网络
    '''

    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, action_dim)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = F.softmax(self.fc2(x), dim=-1)
        return out


class Critic(nn.Module):
    '''
    评论家Critic网络
    '''

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 1)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out


class Actor_Critic(object):
    def __init__(self, env, gamma, lr_a, lr_c, device):
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c

        self.env = env
        self.action_dim = self.env.action_space.n  # 获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  # 获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)  # 创建演员网络
        self.critic = Critic(self.state_dim)  # 创建评论家网络

        self.actor.to(device)
        self.critic.to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss(reduction="sum")

        self.device = device

    def get_action(self, s):
        a = self.actor(s)
        if torch.isnan(a)[0] or torch.isnan(a)[1]:
            a = torch.nan_to_num(a, nan=1e-7, posinf=1e5, neginf=-1e5)
        dist = Categorical(a)
        action = dist.sample()  # 可采取的action
        log_prob = dist.log_prob(action)  # 每种action的概率

        return action.clone(), log_prob

    def learn(self, log_prob_s, s_s, done, rew_s, entropy_coef):
        # 使用Critic网络估计状态值

        terminal = s_s.pop()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        # numpy input of nn

        s_s = torch.tensor(s_s, device=self.device)

        # V

        v = self.critic(s_s).squeeze()

        # reduction matrix

        r_m = torch.tril(
            torch.outer(
                torch.tensor(self.gamma, device=self.device)
                ** torch.arange(0, len(rew_s), step=1, device=self.device),
                torch.tensor(1. / self.gamma, device=self.device)
                ** torch.arange(0, len(rew_s), step=1, device=self.device)
            )
        )
        # Q(include the last state)

        q = torch.mv(
            torch.transpose(r_m, 0, 1), torch.tensor(rew_s, dtype=torch.float32, device=self.device)
        )

        if not done:
            q += self.critic(torch.tensor(terminal, device=self.device)).squeeze() * self.gamma ** len(rew_s)
        # critic loss

        critic_loss = self.loss(q, v)

        # advantage

        advantage = q - v

        # entropy

        entropy = -torch.sum(
            self.actor(s_s) * torch.log(self.actor(s_s))
        )

        # actor loss

        loss_actor = -torch.sum(
            log_prob_s * advantage.detach()
        ) - entropy_coef * entropy

        critic_loss.backward()
        loss_actor.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=4)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=8)

        self.critic_optim.step()
        self.actor_optim.step()
