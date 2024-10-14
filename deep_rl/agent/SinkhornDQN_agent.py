#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch.nn.functional

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *
from main import args

class SinkhornRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        if self.config.multi == 1: # [bs, a, d, N]
            q_values = prediction['sample'].mean(axis=(-2, -1)) # [bs, a]
        else:
            q_values = prediction['sample'].mean(-1)
        return to_np(q_values)



class SinkhornDQNAgent(DQNAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = SinkhornRegressionDQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)

        self.epsilon = config.epsilon
        self.niter_sink = config.niter_sink

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)

        # NEW
        q_, feature = self.network(state)
        if self.config.multi == 1:
            q = q_['sample'].mean(axis=(-2, -1))
        else:
            q = q_['sample'].mean(-1)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return [action], feature


    def sinkhorn_loss(self, x, y, epsilon, n, niter):
        """
        Given two emprical measures with n points each with locations x and y
        outputs an approximation of the OT cost with regularization parameter epsilon
        niter is the max. number of steps in sinkhorn loop
        """
        # The Sinkhorn algorithm takes as input three variables :
        C = self.cost_matrix(x, y, p=args.p).cuda(args.gpu)  # Wasserstein cost function [bs, N, N]
        bs = C.shape[0]  # 32
        # both marginals are fixed with equal weights
        mu = 1. / n * torch.ones(bs, n).cuda(args.gpu)
        nu = 1. / n * torch.ones(bs, n).cuda(args.gpu)
        mu.requires_grad = False
        nu.requires_grad = False

        # Parameters of the Sinkhorn algorithm.
        rho = 1  # (.5) **2          # unbalanced transport
        tau = -.8  # nesterov-like acceleration
        lam = rho / (rho + epsilon)  # Update exponent
        thresh = 10 ** (-1)  # stopping criterion

        # Elementary operations .....................................................................
        def ave(u, u1):
            "Barycenter subroutine, used by kinetic acceleration through extrapolation."
            return tau * u + (1 - tau) * u1

        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

        def lse(A):
            "log-sum-exp"
            return torch.log(torch.exp(A).sum(2, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * mu, 0. * nu, 0.
        actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

        for i in range(niter):
            u1 = u  # useful to check the update
            u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
            v = epsilon * (torch.log(nu) - lse(M(u, v).permute(0, 2, 1)).squeeze()) + v
            err = (u - u1).abs().sum()
            actual_nits += 1
            if (err < thresh).data.cpu().numpy():
                break
        U, V = u, v
        Gamma = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
        cost = torch.sum(Gamma * C)  # Sinkhorn cost
        return cost  # singe element

    def GaussianKernal(self, v1, v2, sigma):  # v1/v2: [bs, N]
        if self.config.multi == 0:  # v: [bs, N]
            v1 = v1.unsqueeze(2)  # [bs, N] -> [bs, N, 1]
            v2 = v2.unsqueeze(1)  # [bs, N] -> [bs, 1, N]
            d = (v1 - v2) ** 2  # [bs, N, N]
        else:  # v: [bs, d, N]
            v1, v2 = v1.permute((0, 2, 1)), v2.permute((0, 2, 1))  # [bs, N, d]
            d = torch.cdist(v1, v2, p=2)  # cdist: for two batches, pdist: within 1 batch
        sigma = 1.0 / torch.tensor(sigma).float().view(-1, 1).cuda(args.gpu)
        temp = torch.matmul(sigma, d.view(1, -1))  # [k, 1] [1, bs*N*N]
        return torch.sum(torch.exp(-temp), dim=0).reshape(v1.shape[0], v1.shape[1], v1.shape[1])  # [bs, N, N]

    def cost_matrix(self, x, y, p=2):  # [bs, N, 1] -> [bs, N, N]
        if self.config.multi == 1:
            # x/y: [bs, d, N]
            x_, y_ = x.permute((0, 2, 1)), y.permute((0, 2, 1))
            c = torch.cdist(x_, y_, p=p) # [bs, N, N]

        else: # one-dimensional setting
            if p == 0:  # Gaussian kernel
                x = y.squeeze(2)  # [bs, N]
                y = y.squeeze(2)  # [bs, N]
                Sigma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                c = self.GaussianKernal(x, y, Sigma)
            # "Returns the matrix of $|x_i-y_j|^p$."
            # unrectified kernel with p
            else:
                x_col = x.unsqueeze(2)  # [bs, N, dimension=1] -> [bs, N, 1, dimension=1]
                y_lin = y.unsqueeze(1)  # [bs, N, dimension=1] -> [bs, 1, N, dimension=1]
                c = torch.sum((torch.abs(x_col - y_lin)) ** p, 3)  # sum over p [bs,N,N]
        return c

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        # samples: [bs, A, N],
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


        ############# Sinkhorn loss
        # x, y: [bs, N] rather than [batch, A, N]
        x, y = x.unsqueeze(2), y.unsqueeze(2) # [bs, N, p=1]
        Wxy = self.sinkhorn_loss(x, y, self.epsilon, self.config.num_samples, self.niter_sink)
        Wxx = self.sinkhorn_loss(x, x, self.epsilon, self.config.num_samples, self.niter_sink)
        Wyy = self.sinkhorn_loss(y, y, self.epsilon, self.config.num_samples, self.niter_sink)
        sink_loss = 2 * Wxy - Wxx - Wyy

        return sink_loss

    def compute_loss_multi(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        # new
        samples_, _ = self.target_network(next_states)
        samples_next = samples_['sample'].detach()
        a_next = torch.argmax(samples_next.sum(axis=(-1, -2)), dim=-1)
        samples_next = samples_next[self.batch_indices, a_next, :, :] # [bs, d, N]

        # decompose the reward
        rewards_mul = self.decompose(transitions.reward)
        rewards_mul = self.config.reward_normalizer(rewards_mul.cpu()).to(self.config.DEVICE) # original, decompose, normalize
        rewards = tensor(rewards_mul).unsqueeze(-1) # [bs, d, 1]
        masks = tensor(transitions.mask).unsqueeze(-1).unsqueeze(-1) # [bs, 1, 1]
        samples_next = rewards + self.config.discount ** self.config.n_step * masks * samples_next

        samples_, _ = self.network(states)
        samples = samples_['sample']

        actions = tensor(transitions.action).long()
        samples = samples[self.batch_indices, actions, :, :]
        x, y = samples, samples_next # [batch, d, N], e.g., N=200


        ############# Sinkhorn loss
        # x, y: [bs, d, N]
        Wxy = self.sinkhorn_loss(x, y, self.epsilon, self.config.num_samples, self.niter_sink)
        Wxx = self.sinkhorn_loss(x, x, self.epsilon, self.config.num_samples, self.niter_sink)
        Wyy = self.sinkhorn_loss(y, y, self.epsilon, self.config.num_samples, self.niter_sink)
        sink_loss = 2 * Wxy - Wxx - Wyy

        return sink_loss

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