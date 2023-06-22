# pytorch-a3c

This is a PyTorch implementation of Asynchronous Advantage Actor Critic (A3C) and a toy example on the CartPole-v1 from ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf).

This implementation is inspired by [Universe Starter Agent](https://github.com/openai/universe-starter-agent) and ikostrikov's implementation of A3C(https://github.com/ikostrikov/pytorch-a3c/tree/master).

The test.py in the example folder is useless, just for debugging.

According to my experience, if the example is trained on the server , "torch.multiprocessing.set_start_method("spawn")" should be add at the top of the main.py. 



