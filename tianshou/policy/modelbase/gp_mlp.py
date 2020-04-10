import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import BasePolicy
from tianshou.utils.visom import VisdomLinePlotter
# from tianshou.exploration import OUNoise


class GP_MLPPolicy(BasePolicy):
    """Implementation of an easy model-based PG algorithm using GP as the
    dynamic model

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module dynamic model: the model network. (s, a -> netx_s)
    :param torch.optim.Optimizer model_optim: the optimizer for model
        network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param action_range: the action range (minimum, maximum).
    :type action_range: [float, float]
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.
    """

    def __init__(self, actor, actor_optim, model, model_optim, cost_fn,
                 tau=0.005, exploration_noise=0.1,
                 action_range=None, reward_normalization=False,
                 ignore_done=False,  horizon=40, **kwargs):
        super().__init__(**kwargs)
        if actor is not None:
            self.actor, self.actor_old = actor, deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim = actor_optim
        if model is not None:
            self.model = model
            self.model.eval()
            self.model.set_optimizer(model_optim)
        assert cost_fn is not None, 'cost_fn should not be none'
        self.cost_fn = cost_fn
        assert 0 <= tau <= 1, 'tau should in [0, 1]'
        self._tau = tau
        assert 0 <= exploration_noise, 'noise should not be negative'
        self._eps = exploration_noise
        assert action_range is not None
        self._range = action_range
        self._action_bias = (action_range[0] + action_range[1]) / 2
        self._action_scale = (action_range[1] - action_range[0]) / 2
        # it is only a little difference to use rand_normal
        # self.noise = OUNoise()
        self._rm_done = ignore_done
        self._rew_norm = reward_normalization
        self.__eps = np.finfo(np.float32).eps.item()
        self.horizon = horizon
        self.plotter = VisdomLinePlotter()

    def set_eps(self, eps):
        """Set the eps for exploration."""
        self._eps = eps

    def train(self):
        """Set the module in training mode, except for the target network."""
        self.training = True
        self.actor.train()

    def eval(self):
        """Set the module in evaluation mode, except for the target network."""
        self.training = False
        self.actor.eval()

    def sync_weight(self):
        """Soft-update the weight for the target network."""
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def process_fn(self, batch, buffer, indice):
        if self._rew_norm:
            bfr = buffer.rew[:min(len(buffer), 1000)]  # avoid large buffer
            mean, std = bfr.mean(), bfr.std()
            if std > self.__eps:
                batch.rew = (batch.rew - mean) / std
        if self._rm_done:
            batch.done = batch.done * 0.
        return batch

    def __call__(self, batch, state=None,
                 model='actor', input='obs', eps=None, **kwargs):
        """Compute action over the given batch data.

        :param float eps: in [0, 1], for exploration use.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        More information can be found at
        :meth:`~tianshou.policy.BasePolicy.__call__`.
        """
        model = getattr(self, model)
        obs = getattr(batch, input)
        logits, h = model(obs, state=state, info=batch.info)
        logits += self._action_bias
        if eps is None:
            eps = self._eps
        if eps > 0:
            # noise = np.random.normal(0, eps, size=logits.shape)
            # logits += torch.tensor(noise, device=logits.device)
            # noise = self.noise(logits.shape, eps)
            logits += torch.randn(
                size=logits.shape, device=logits.device) * eps
        logits = logits.clamp(self._range[0], self._range[1])
        return Batch(act=logits, state=h)

    def learn(self, batch, batch_size=None, repeat=1, **kwargs):
        model_losses, actor_losses = [], []
        # train dynamic model
        self.model.collect_data(batch)
        m_loss = self.model.optimize(repeat=20)

        for _ in range(repeat):
            a_loss = 0
            obs_next = torch.tensor(
                batch.obs[0], dtype=torch.float, device=self.model.device)[None, :]
            self.actor_optim.zero_grad()
            real_obs, predict_obs = [], []
            for i in range(min(self.horizon, batch_size)):
                obs_next = self.model(obs_next, self(
                    Batch(obs=obs_next, info={})).act)
                a_loss += self.cost_fn(obs_next)
                real_obs.append(batch.obs_next[i])
                predict_obs.append(obs_next[0].data.cpu().numpy())

            a_loss.backward()
            self.actor_optim.step()
            actor_losses.append(a_loss.item())
            model_losses.append(m_loss.item())

        self._plot(np.array(real_obs), np.array(predict_obs))
        self.sync_weight()
        return {
            'loss/actor': actor_losses,
            'loss/model': model_losses,
        }

    def _plot(self, real_obs, predict_obs):
        assert real_obs.shape == predict_obs.shape, "shape should be identical"
        for d in range(real_obs.shape[-1]):
            self.plotter.plot(f'multi_dim={d:d}', 'true',
                              f'Model Multi-Step Prediction dim={d:d}',
                              range(real_obs.shape[0]),
                              real_obs[:, d], update='replace')
            self.plotter.plot(f'multi_dim={d:d}', 'predict',
                              f'Model Multi-Step Prediction dim={d:d}',
                              range(predict_obs.shape[0]),
                              predict_obs[:, d], update='replace')
