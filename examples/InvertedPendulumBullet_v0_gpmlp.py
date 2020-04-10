import gym
import pybullet_envs
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy.modelbase.gp_mlp import GP_MLPPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import VectorEnv, SubprocVectorEnv

from continuous_net import Actor
from tianshou.model.mgpr import GPDynamicsModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        default='InvertedPendulumBulletEnv-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--model-lr', type=float, default=1e-2)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=5)
    parser.add_argument('--collect-per-step', type=int, default=5)
    parser.add_argument('--repeat-per-collect', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=2)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=1/35)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_gpmlp(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # train_envs = gym.make(args.task)
    train_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    actor = Actor(
        args.layer_num, args.state_shape, args.action_shape,
        args.max_action, args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    model = GPDynamicsModel(
        args.state_shape, args.action_shape, args.device
    ).to(args.device)
    model_optim = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    def obs_cost_fn(obs):
        target = torch.FloatTensor([0,0,1,0,0])
        c = (obs - target)**2
        c = -c.sum(dim=1)
        return -c.exp()

    # log
    writer = SummaryWriter(args.logdir + '/' + 'gpmlp')

    policy = GP_MLPPolicy(
        actor, actor_optim, model, model_optim, obs_cost_fn,
        args.tau, args.exploration_noise,
        [env.action_space.low[0], env.action_space.high[0]],
        reward_normalization=True, ignore_done=True)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, stop_fn=stop_fn, writer=writer,
        task=args.task)
    # assert stop_fn(result['best_reward'])
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_gpmlp()
