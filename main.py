import os, torch
from deep_rl import *
import argparse
import gym

parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('--game', default="Asteroids", type=str)
parser.add_argument('--method', default="MMD", type=str, help='DQN, QRDQN, C51, Sinkhorn, MMD')
parser.add_argument('--iter', default=1e7, type=int, help='number of iterations')
parser.add_argument('--seed', default=1, type=int, help='1,2,3')
parser.add_argument('--evaluate', default=0, type=int, help='0: train, 1: evaluate')
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--niter_sink', default=10, type=int)
parser.add_argument('--epsilon', default=10.0, type=float)
parser.add_argument('--samples', default=200, type=int)
parser.add_argument('--p', default=2, type=int)
parser.add_argument('--multi', default=0, type=int, help='0: 1D return distribution, 1: multi-dimensional reward/return distribution')
args = parser.parse_args()
print(args)

if args.gpu != -1:
    Number_thread = 8
    os.environ["MKL_NUM_THREADS"] = str(Number_thread)
    os.environ["NUMEXPR_NUM_THREADS"] = str(Number_thread)
    os.environ["OMP_NUM_THREADS"] = str(Number_thread)
    torch.set_num_threads(Number_thread)

# (1) DQN
def dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, seed=args.seed)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    # config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
    config.batch_size = 32
    config.discount = 0.99
    config.history_length = 4
    config.max_steps = int(args.iter)
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length,
    )
    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    # config.exploration_steps = 100
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = False
    config.async_actor = True

    # new parameters
    config.game_file = args.game

    config.logtxtname = args.method + str(int(args.iter)) + '_seed' + str(args.seed)

    Agent = DQNAgent(config)
    if args.evaluate == 1:  # normal training and save model
        print('Action dim is ', config.action_dim)
        run_steps_evaluate(Agent)
    else:
        run_steps(Agent) # main function


# (2) QR-DQN
def quantile_regression_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, seed=args.seed)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda: QuantileNet(config.action_dim, config.num_quantiles, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.batch_size = 32
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        history_length=4,
    )
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=True)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.num_quantiles = 200
    config.max_steps = int(args.iter)

    # new parameters
    config.game_file = args.game
    config.logtxtname = args.method + str(int(args.iter)) + '_seed' + str(args.seed)

    Agent = QuantileRegressionDQNAgent(config)
    if args.evaluate == 1:  # normal training and save model
        print('Action dim is ', config.action_dim)
        run_steps_evaluate(Agent)
    else:
        run_steps(Agent)  # main function

# (3) C51
def categorical_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, seed=args.seed)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.batch_size = 32
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        history_length=4,
    )
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=True)

    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    config.sgd_update_frequency = 4
    config.gradient_clip = 0.5
    config.max_steps = int(args.iter)

    # new parameters
    config.game_file = args.game
    config.logtxtname = args.method + str(int(args.iter)) + '_seed' + str(args.seed)

    Agent = CategoricalDQNAgent(config)
    if args.evaluate == 1:  # normal training and save model
        print('Action dim is ', config.action_dim)
        run_steps_evaluate(Agent)
    else:
        run_steps(Agent)  # main function

# (4) Sinkhorn-DQN
def Sinkhorn_regression_dqn_pixel(**kwargs):
    print('running sinkhorn dqn')
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.task_fn = lambda: Task(config.game, seed=args.seed)
    config.eval_env = config.task_fn()
    config.num_samples = args.samples
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    # config.optimizer_fn_phi = lambda params: torch.optim.Adam(params, lr=0.001, eps=0.01 / 32)


    if args.multi == 1:
        # read the file: Asterix, Asteroids, MsPacman, Gopher, UpNDown, pong,
        with open('reward-compose/{}-reward.txt'.format(args.game), 'r') as file:
            text = file.read()
            config.rewards = eval(text)
            config.reward_dim = len(config.rewards)
        config.network_fn = lambda: SinkNet_multi(config.action_dim, config.num_samples, config.reward_dim, NatureConvBody())
        config.reward_normalizer = ClipNormalizer()
    else:
        config.network_fn = lambda: SinkNet(config.action_dim, config.num_samples, NatureConvBody())
        config.reward_normalizer = SignNormalizer()

    # config.network_fn_phi = lambda: PhiNet() # cost embedding network
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.batch_size = 32
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        history_length=4,
    )
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=True)

    config.state_normalizer = ImageNormalizer()

    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.max_steps = int(args.iter)

    # new parameters
    config.game_file = args.game
    name_multi = 'multi' if args.multi == 1 else ''
    config.multi = args.multi


    ######### change p for the sensitivity analysis
    if args.p == 2:
        config.logtxtname = args.method + name_multi + str(int(args.iter)) + '_epsilon' + str(args.epsilon) + '_iter' + str(args.niter_sink) + '_sample'+str(args.samples)  + '_seed' + str(args.seed)
    else:
        config.logtxtname = args.method + name_multi + str(int(args.iter)) + '_epsilon' + str(args.epsilon) + '_iter' + str(args.niter_sink) + '_sample'+str(args.samples) + '_p'+ str(args.p) + '_seed' + str(args.seed)

    config.niter_sink = args.niter_sink
    config.epsilon = args.epsilon

    Agent = SinkhornDQNAgent(config)
    if args.evaluate == 1:  # normal training and save model
        print('Action dim is ', config.action_dim)
        run_steps_evaluate(Agent)
    else:
        run_steps(Agent)  # main function


# (5) MMD: similar to Sinkhorn
def MMD_dqn_pixel(**kwargs):
    print('running MMD DQN')
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, seed=args.seed)
    config.eval_env = config.task_fn()
    config.num_samples = args.samples
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)

    if args.multi == 1:
        # read the file: Asterix, Asteroids, MsPacman, Gopher, UpNDown, pong,
        with open('reward-compose/{}-reward.txt'.format(args.game), 'r') as file:
            text = file.read()
            config.rewards = eval(text)
            config.reward_dim = len(config.rewards)
        config.network_fn = lambda: MMDNet_multi(config.action_dim, config.num_samples, config.reward_dim, NatureConvBody())  # difference
        config.reward_normalizer = ClipNormalizer() # same as the original Neurips paper, but different from original code
    else:
        config.network_fn = lambda: MMDNet(config.action_dim, config.num_samples, NatureConvBody())  # difference
        config.reward_normalizer = SignNormalizer()
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.batch_size = 32
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        history_length=4,
    )
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=True)
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    # config.exploration_steps = 2000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.max_steps = int(args.iter)

    # new parameters
    config.game_file = args.game
    if args.multi == 1:
        config.logtxtname = args.method + 'multi'+ str(int(args.iter)) + '_seed' + str(args.seed)
    else:
        config.logtxtname = args.method + str(int(args.iter)) + '_seed' + str(args.seed)
    config.multi = args.multi

    Agent = MMDAgent(config) # difference
    if args.evaluate == 1:  # normal training and save model
        print('Action dim is ', config.action_dim)
        run_steps_evaluate(Agent)
    else:
        run_steps(Agent)  # main function



if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    if args.gpu == -1:
        set_one_thread()
    random_seed(seed=args.seed)
    select_device(args.gpu)  # -1: GPU, 0-8: GPU id

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('log/' + args.game):
        os.makedirs('log/' + args.game)
    if not os.path.exists('model/'+args.game):
        os.makedirs('model/'+args.game)

    if args.game == 'breakout':
        game = 'BreakoutNoFrameskip-v4'
    elif args.game == 'spaceinvader':
        game = 'SpaceInvadersNoFrameskip-v4'
    elif args.game == 'qbert':
        game = 'QbertNoFrameskip-v4'
    elif args.game == 'enduro':
        game = 'EnduroNoFrameskip-v4'
    elif args.game == 'pong':
        game = 'PongNoFrameskip-v4'
    else: # YarsRevenge, UpNDown, Robotank
        game = args.game+'NoFrameskip-v4' # other 3 games

    if args.method == 'DQN':
        dqn_pixel(game=game)
    elif args.method == 'QRDQN':
        quantile_regression_dqn_pixel(game=game)
    elif args.method == 'C51':# C51
        categorical_dqn_pixel(game=game)
    elif args.method == 'Sinkhorn':
        Sinkhorn_regression_dqn_pixel(game=game)
    elif args.method == 'MMD':
        MMD_dqn_pixel(game=game)
    else:
        print('method name is a mistake!')
