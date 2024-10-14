#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *
# from geomloss import SamplesLoss
from main import args

class MMDActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        if self.config.multi == 1: # [bs, a, d, N]
            q_values = prediction['sample'].mean(axis=(-2, -1)) # [bs, a]
        else:
            q_values = prediction['sample'].mean(-1)
        return to_np(q_values)


class MMDAgent(DQNAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = MMDActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        # ew
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        # q= self.network(state)['quantile'].mean(-1)
        # NEW
        q_, feature = self.network(state)
        if self.config.multi == 1:
            q = q_['sample'].mean(axis=(-2, -1)) # algorithm 1 in Neurips paper
        else:
            q = q_['sample'].mean(-1)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return [action], feature

    def GaussianKernal(self, v1, v2, sigma):  # v1/v2: [bs, N]
        if self.config.multi == 0: # v: [bs, N]
            v1 = v1.unsqueeze(2)  # [bs, N] -> [bs, N, 1]
            v2 = v2.unsqueeze(1) # [bs, N] -> [bs, 1, N]
            d = (v1 - v2) ** 2 # [bs, N, N]
        else:  # v1/v2: [bs, d, N]
            # change the dimension [bs, d, N] -> [bs, N, d]
            # v1 = v1.permute((0, 2, 1))
            # batch mm
            # d = torch.bmm(v1, v2) # [bs, N, d] * [bs, d, N] -> [bs, N, N]
            v1, v2 = v1.permute((0, 2, 1)), v2.permute((0, 2, 1))  # [bs, N, N]
            d = torch.cdist(v1, v2, p=2) # cdist: for two batches, pdist: within 1 batch

        sigma = 1.0 / torch.tensor(sigma).float().view(-1, 1).cuda(args.gpu)
        temp = torch.matmul(sigma, d.view(1, -1))  # [k, 1] [1, bs*N*N]
        return torch.sum(torch.exp(-temp), dim=0).reshape(v1.shape[0], v1.shape[1], v1.shape[1]) # [bs, N, N]

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        # new
        samples_, _ = self.target_network(next_states)
        samples_next = samples_['sample'].detach()
        a_next = torch.argmax(samples_next.sum(-1), dim=-1)
        samples_next = samples_next[self.batch_indices, a_next, :] # Z(s',a*) = [bs, N]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        samples_next = rewards + self.config.discount ** self.config.n_step * masks * samples_next

        samples_, _ = self.network(states)
        samples = samples_['sample']

        actions = tensor(transitions.action).long()
        samples = samples[self.batch_indices, actions, :]
        x, y = samples, samples_next # [batch, N], e.g., N=200

        ############# MMD (mixed kernel)
        Sigma = [1,2,3,4,5,6,7,8,9,10]
        xixj = self.GaussianKernal(x, x, Sigma)
        yiyj = self.GaussianKernal(y, y, Sigma)
        xiyj = self.GaussianKernal(x, y, Sigma)
        MMD_loss = xixj + yiyj - 2 * xiyj

        return MMD_loss

    def compute_loss_multi(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        # new
        samples_, _ = self.target_network(next_states) # [bs, a, N]
        samples_next = samples_['sample'].detach()
        # MMD: [bs, a, N] -> [bs, a] -> find max; MMD with multiple D: [bs, a, d, N] -> [bs, a] (Algorithm 1 in paper) -> find max
        a_next = torch.argmax(samples_next.sum(axis=(-1, -2)), dim=-1)
        samples_next = samples_next[self.batch_indices, a_next, :, :] # [bs, d, N]

        # decompose the reward

        rewards_mul = self.decompose(transitions.reward) # [bs, d]
        rewards_mul = self.config.reward_normalizer(rewards_mul.cpu()).to(self.config.DEVICE) #### original reward, decompose, then normalize
        rewards = tensor(rewards_mul).unsqueeze(-1) # [bs, d, 1]
        masks = tensor(transitions.mask).unsqueeze(-1).unsqueeze(-1) # [bs, 1, 1]
        # print('reward: ', rewards.shape) # [32, 3, 1]
        samples_next = rewards + self.config.discount ** self.config.n_step * masks * samples_next # [bs, 1, 1] * [bs, d, N] -> [bs, d, N]
        # print('samples: ', samples_next.shape) # [32, 3, 200]

        # current distribution
        samples_, _ = self.network(states)
        samples = samples_['sample']
        actions = tensor(transitions.action).long()
        samples = samples[self.batch_indices, actions, :, :]

        assert rewards.shape[1] == self.config.reward_dim
        assert samples.shape == (rewards.shape[0], self.config.reward_dim, self.config.num_samples)
        assert samples_next.shape == (rewards.shape[0], self.config.reward_dim, self.config.num_samples)
        x, y = samples, samples_next # [bs, d, N] e.g., N=200

        ############# MMD (mixed kernel)
        Sigma = [1,2,3,4,5,6,7,8,9,10]
        xixj = self.GaussianKernal(x, x, Sigma)
        yiyj = self.GaussianKernal(y, y, Sigma)
        xiyj = self.GaussianKernal(x, y, Sigma)
        MMD_loss = xixj + yiyj - 2 * xiyj
        return MMD_loss


    def reduce_loss(self, loss):
        return loss.mean()

    def step(self):
        config = self.config
        transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            if self.config.multi == 1:
                Rewardss = [r for r in rewards]
            else:
                Rewardss = [config.reward_normalizer(r) for r in rewards]
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                # reward=[config.reward_normalizer(r) for r in rewards],
                reward=Rewardss,
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))


        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()
            if config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()

            if self.config.multi == 1:
                loss = self.compute_loss_multi(transitions)
            else:
                loss = self.compute_loss(transitions)
            if isinstance(transitions, PrioritizedTransition):
                priorities = loss.abs().add(config.replay_eps).pow(config.replay_alpha)
                idxs = tensor(transitions.idx).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(transitions.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
                weights = weights / weights.max()
                loss = loss.mul(weights)

            loss = self.reduce_loss(loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step() # quantile network


        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
