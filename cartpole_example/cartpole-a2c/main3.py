import gym
from model3 import Actor_Critic
import wandb
import torch
from cartpole_example.config import Config
import time


def coef_test(config, run_handle, device):
    t1 = time.time()

    def query_environment(name):
        env = gym.make(name)
        spec = gym.spec(name)
        print(f"Action Space: {env.action_space}")
        print(f"Observation Space: {env.observation_space}")
        print(f"Max Episode Steps: {spec.max_episode_steps}")
        print(f"Nondeterministic: {spec.nondeterministic}")
        print(f"Reward Range: {env.reward_range}")
        print(f"Reward Threshold: {spec.reward_threshold}")

    query_environment('CartPole-v1')
    env = gym.make('CartPole-v1')
    model = Actor_Critic(env, config.gamma, config.lr_a, config.lr_c, device)  # 实例化Actor_Critic算法类
    reward = []
    for episode in range(config.training_episode):
        s_s = [env.reset(seed=0)[0]]  # 获取环境状态
        log_prob_s = torch.tensor([], device=device)
        rew_s = []
        done = False     # 记录当前回合游戏是否结束
        ep_r = 0
        step = 0

        while not done and step < config.max_step:
            # 通过Actor_Critic算法对当前环境做出行动
            a, log_prob = model.get_action(torch.tensor(s_s[-1], device=device))
            log_prob_s = torch.cat((log_prob_s, log_prob.unsqueeze(dim=0)))
            # 获得在做出a行动后的最新环境
            s_, rew, done, _, _ = env.step(a.cpu().numpy())
            s_s.append(s_)
            rew_s.append(rew)

            # 计算当前reward
            ep_r += rew
            step += 1

        # 训练模型
        model.learn(log_prob_s, s_s, done, rew_s, config.entropy_coef)

        # 显示奖励
        reward.append(ep_r)
        run_handle.log({"episode_reward": ep_r}, step=episode)
        print(f"episode:{episode} ep_r:{ep_r}")
    t2 = time.time()
    run_time = t2 - t1
    run_handle.log({"running time": run_time / 60})


if __name__ == "__main__":
    seed = int(input("the seed is : "))
    device_name = input('the device is : ')
    device = torch.device(device_name)
    config_ = Config(
        gamma=0.99,
        lr_a=2e-4,
        lr_c=2e-5,
        training_episode=2000,
        entropy_coef=1e-3,
        max_step=300,
    )
    Run_handle = wandb.init(
        project="2023_6_24_cartpole_no_parallel",
        config={
            "seed": seed,
            "episode_num": config_.training_episode,
            "entropy_coefficient": config_.entropy_coef,
            "gamma": config_.gamma,
            'lr_a': config_.lr_a,
            'lr_c': config_.lr_c,
            'max_step': config_.max_step
        }
    )
    coef_test(config_, Run_handle, device)


