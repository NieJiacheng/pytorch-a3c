import gym
from model2 import Actor_Critic
import wandb
from torch import multiprocessing as mp
from torch import device, tensor, cat, manual_seed, cuda
from ctypes import c_int
from config import Config
import numpy.random as np_random


def set_seed(seed):
    np_random.seed(seed)
    manual_seed(seed)  # torch seed
    cuda.manual_seed(seed)


class TrainProcess(mp.Process):
    def __init__(
            self,
            p_id,
            env_name,
            config,
            device,
            global_actor_critic,
            i_episode,
            lock_wandb,
            run_handle
    ):
        super().__init__(name="process_%d" % p_id)
        self.env = gym.make(env_name)
        self.config = config
        self.ActorCritic = Actor_Critic(self.env, self.config.gamma, self.config.lr_a, self.config.lr_c, device)
        self.global_actor_critic = global_actor_critic
        self.i_episode = i_episode
        self.lock_wandb = lock_wandb
        self.run_handle = run_handle
        self.device = device

    def sync_network(self, local_model, shared_model):
        local_model.load_state_dict(shared_model.state_dict())

    def run(self):
        while self.i_episode.value < self.config.training_episode:
            state_buffer = [self.env.reset(seed=0)[0]]
            reward_buffer = []

            # catch the original log with grad information.
            # list here didn't work since transfer list into tensor will drop teh grad information.
            log_prob_tensor_buffer = tensor([], device=self.device)

            is_done = False
            episode_reward = 0

            self.sync_network(self.ActorCritic.actor, self.global_actor_critic.actor)
            self.sync_network(self.ActorCritic.critic, self.global_actor_critic.critic)

            while not is_done:
                action, log_prob = self.ActorCritic.get_action(tensor(state_buffer[-1], device=self.device))

                # log_prob: tensor(x) ----> tensor([x])
                log_prob_tensor_buffer = cat((log_prob_tensor_buffer, log_prob.unsqueeze(dim=0)))

                state_next, pre_state_reward, is_done, _, _ = self.env.step(action.cpu().numpy())
                state_buffer.append(state_next)
                reward_buffer.append(pre_state_reward)
                episode_reward += pre_state_reward
            self.i_episode.value += 1

            with self.lock_wandb:
                self.run_handle.log({"episode_reward": episode_reward}, step=self.i_episode.value)
                print(f"episode:{self.i_episode.value} ep_r:{episode_reward}")

            self.ActorCritic.actor_optim.zero_grad()
            self.ActorCritic.critic_optim.zero_grad()

            self.ActorCritic.learn_for_global(
                state_buffer,
                is_done,
                reward_buffer,
                log_prob_tensor_buffer,
                self.config.entropy_coef,
                self.global_actor_critic,
            )


if __name__ == "__main__":
    # for server training
    # mp.set_start_method("spawn")
    def query_environment(name):
        env = gym.make(name)
        spec = gym.spec(name)
        print(f"Action Space: {env.action_space}")
        print(f"Observation Space: {env.observation_space}")
        print(f"Max Episode Steps: {spec.max_episode_steps}")
        print(f"Nondeterministic: {spec.nondeterministic}")
        print(f"Reward Range: {env.reward_range}")
        print(f"Reward Threshold: {spec.reward_threshold}")
        return env

    config = Config(
        gamma=0.99,
        lr_a=2e-4,
        lr_c=2e-5,
        training_episode=2000,
        entropy_coef=1e-3
    )

    Run_handle = wandb.init(
        project="2023_6_22_cartpole_parallel",
        config={"episode_num": config.training_episode, "entropy_coefficient": config.entropy_coef}
    )

    env = query_environment('CartPole-v1')

    seed = int(input("the seed of the experiment is : "))
    set_seed(seed)

    jobs = input("the number of processes is : ")

    processes = []

    device_name = input("device : ")
    Device = device(device_name)

    GlobalActorCritic = Actor_Critic(env, config.gamma, config.lr_a, config.lr_c, 'cpu')
    GlobalActorCritic.share_network()

    I_episode = mp.Value(c_int, 0)
    Lock_wandb = mp.RLock()

    for job_id in range(int(jobs)):
        processes.append(
            TrainProcess(
                job_id,
                'CartPole-v1',
                config,
                Device,
                GlobalActorCritic,
                I_episode,
                Lock_wandb,
                Run_handle
            )
        )
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    Run_handle.finish()



