
import torch
import gpytorch
import numpy as np
from tianshou.data import ReplayBuffer
from tianshou.data import Batch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_out = train_y.shape[1]
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.num_out]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1],
                                       # lengthscale_prior = gpytorch.priors.GammaPrior(1,10),
                                       batch_shape=torch.Size([self.num_out])),
            batch_shape=torch.Size([self.num_out]),
            # outputscale_prior = gpytorch.priors.GammaPrior(1.5,2),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class GPDynamicsModel(torch.nn.Module):
    """Gaussian Process Model for learning environment transition."""

    def __init__(
            self,
            observation_shape,
            action_shape,
            device,
            buffer_size=int(1e4),
            data_size=int(5e2)
    ):
        """Instantiate GP according to inputs."""
        super().__init__()

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=int(np.prod(observation_shape)))
        # get 10 random points as placeholder
        self.train_data = Batch(input=torch.rand(
            data_size, int(np.prod(observation_shape) + np.prod(action_shape))),
            output=torch.rand(data_size, int(np.prod(observation_shape))),
            mean=None, std=None)
        self.gp = ExactGPModel(
            self.train_data.input,
            self.train_data.output,
            self.likelihood
        )
        self.data_size = data_size
        self.device = device
        self.buffer = ReplayBuffer(size=buffer_size)
        self.optimizer = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def collect_data(self, batch):
        for data in batch:
            self.buffer.add(obs=data.obs, act=data.act, rew=data.rew,
                            done=data.done, obs_next=data.obs_next, info=data.info)

    def optimize(self, repeat=1):
        # select data here
        self._set_train_data(
            self.buffer[:min(len(self.buffer), self.data_size)])
        assert self.optimizer is not None, "model optimizer should not be None"
        self.gp.train()
        self.likelihood.train()
        mml = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.gp)
        # self.agent.d_model.randomize()
        for _ in range(repeat):
            self.optimizer.zero_grad()
            pred_obs_delta = self.gp(self.train_data.input)
            d_loss = -mml(pred_obs_delta, self.train_data.output)
            d_loss.backward()
            self.optimizer.step()
        # next_obs = torch.clamp(
        #     next_obs, -samples.env.observation_space.high, samples.env.observation_space.high)
        return d_loss

    def _set_train_data(self, data):
        X = np.concatenate([data.obs, data.act], axis=-1)
        Y = data.obs_next - data.obs
        X = torch.tensor(X, dtype=torch.float, device=self.device)
        self.train_data.mean = X.mean()
        self.train_data.std = X.std()
        X = (X - self.train_data.mean)/self.train_data.std
        Y = torch.tensor(Y, dtype=torch.float, device=self.device)
        self.gp.set_train_data(X, Y, strict=False)
        self.train_data.input = X
        self.train_data.output = Y

    def forward(self, obs, act):
        gp_input = torch.cat([obs, act], dim=1)
        gp_input = (gp_input - self.train_data.mean)/self.train_data.std
        self.gp.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            obs_delta = self.likelihood(self.gp(gp_input)).mean.squeeze(-1)
            obs_next = obs_delta + obs

        return obs_next

    def randomize(self):
        mean = 0
        sigma = 1
        with torch.no_grad():
            self.gp.covar_module.base_kernel._set_lengthscale(0)
            self.gp.covar_module._set_outputscale(0)
            self.likelihood._set_noise(0.1)
