class Config(object):
    def __init__(self, gamma, lr_a, lr_c , training_episode, entropy_coef, max_step):
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.training_episode = training_episode
        self.entropy_coef = entropy_coef
        self.max_step = max_step

