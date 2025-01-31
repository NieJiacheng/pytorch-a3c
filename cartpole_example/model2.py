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
        if isinstance(s[0], np.ndarray):
            s = torch.FloatTensor(s[0])
        elif isinstance(s, np.ndarray):
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
        if isinstance(s[0], np.ndarray):
            s = torch.FloatTensor(s[0])
        elif isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out


class Actor_Critic(object):
    def __init__(self, env, gamma, lr_a, lr_c, device):
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.device = device

        self.env = env
        self.action_dim = self.env.action_space.n             # 获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  # 获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   # 创建演员网络
        self.critic = Critic(self.state_dim)  # 创建评论家网络
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss(reduction="sum")

    def get_action(self, s):
        a = self.actor(s)
        if torch.isnan(a)[0] or torch.isnan(a)[1]:
            a = torch.nan_to_num(a, nan=1e-7, posinf=1e5, neginf=-1e5)
        dist = Categorical(a)
        action = dist.sample()             # 可采取的action
        log_prob = dist.log_prob(action)   # 每种action的概率

        return action.clone(), log_prob

    def migrate_grad(self, l_net, g_net):
        l_parms = list(l_net.parameters())
        g_parms = list(g_net.parameters())
        for layer in range(len(l_parms)):
            g_parms[layer].grad = torch.zeros(size=g_parms[layer].size())
            g_parms[layer].grad += l_parms[layer].grad.clone().cpu()

    def learn_for_global(
            self,
            s_buffer,
            done,
            r_buffer,
            log_prob_tensor_buffer,
            entropy_coef,
            g_actor_critic,
            lock_shared_actor,
            lock_shared_critic
    ):
        # terminal of trajectory

        end = s_buffer.pop()

        # buffer_of_state:list(tensor) -------> 1-D CUDA tensor.
        # It's safe since the state information didn't contain any gradient information of network.

        s_buffer = torch.tensor(s_buffer, device=self.device)

        # state value of the trajectory.
        # squeeze() is used to make it in a harmonious form.

        v = self.critic(s_buffer).squeeze()

        # reduction matrix
        '''
        [1,  1/2,   1/4] × [1,   2,   4]
        
        =  [[1 ,    2,   4]            
            [1/2,   1,   2]  
            [1/4, 1/2,   1]]
                  
        -> [[1 ,    0,   0]            
            [1/2,   1,   0]  
            [1/4, 1/2,   1]]
        '''

        r_m = torch.tril(
            torch.outer(
                torch.tensor(self.gamma, device=self.device) ** torch.arange(0, len(r_buffer), step=1, device=self.device),
                torch.tensor(1. / self.gamma, device=self.device) ** torch.arange(0, len(r_buffer), step=1, device=self.device)
            )
        )

        # sum of reward with reduction
        '''
                                  [[1 ,    0,   0]            
          [r1, r2 , r3]      *     [1/2,   1,   0]          = [r1 + r2 / 2 + r3 / 4, r2 + r3 / 2, r3]
                                   [1/4, 1/2,   1]].T.T
        '''

        q = torch.mv(
            torch.transpose(r_m, 0, 1), torch.tensor(r_buffer, dtype=torch.float32, device=self.device)
        )

        # r1 + r2 / 2 + r3 / 4 + V(terminal) / 8

        if not done:
            q += self.critic(torch.tensor(end, device=self.device)).squeeze() * self.gamma ** len(r_buffer)

        # loss of critic : advantage ** 2

        critic_loss = self.loss(q, v)

        # advantage

        advantage = q - v

        # entropy

        entropy = -torch.sum(
            self.actor(s_buffer) * torch.log(self.actor(s_buffer))
        )

        # loss of actor:  log(pi) * advantage_ + entropy * coefficient

        actor_loss = -torch.sum(
            log_prob_tensor_buffer * advantage.detach()
        ) - entropy_coef * entropy

        actor_loss.backward()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=4)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=8)

        # migrate gradient from local network to global network after set grad of global network to "zero"

        with lock_shared_actor:
            self.migrate_grad(self.actor, g_actor_critic.actor)
            g_actor_critic.actor_optim.step()

        with lock_shared_critic:
            self.migrate_grad(self.critic, g_actor_critic.critic)
            g_actor_critic.critic_optim.step()



    def share_network(self):
        self.actor.share_memory()
        self.critic.share_memory()


if __name__ == "__main__":
    from torchviz import make_dot
    a = Actor(4, 4)
    t = torch.tensor([1., 2., 1., .5])
    g = make_dot(a(t), params=dict(list(a.named_parameters())), show_attrs=True)
    g.render("g3", "D:\OneDrive - CUHK-Shenzhen\桌面", view=True)

