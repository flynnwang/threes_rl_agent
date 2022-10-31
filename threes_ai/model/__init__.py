import logging
from typing import Callable, Dict, Optional, Tuple, Union, NamedTuple, Any

import gym
import numpy as np
import torch
from torch import nn
from torchvision.ops import SqueezeExcitation

from torch.distributions.categorical import Categorical

from threes_ai.threes.consts import *


class RewardSpec(NamedTuple):
  reward_min: float
  reward_max: float
  zero_sum: bool


reward_spec = RewardSpec(
    reward_min=0,
    reward_max=1.,
    zero_sum=False,
)


def _index_select(embedding_layer: nn.Embedding,
                  x: torch.Tensor) -> torch.Tensor:
  out = embedding_layer.weight.index_select(0, x.view(-1))
  return out.view(*x.shape, -1)


def _forward_select(embedding_layer: nn.Embedding,
                    x: torch.Tensor) -> torch.Tensor:
  return embedding_layer(x)


def _get_select_func(use_index_select: bool) -> Callable:
  """
    Use index select instead of default forward to possibly speed up embedding.
    NB: This disables padding_idx functionality
    """
  if use_index_select:
    return _index_select
  else:
    return _forward_select


class DictInputLayer(nn.Module):

  @staticmethod
  def forward(
      x: Dict[str, Union[Dict, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return x["obs"]


class ConvEmbeddingInputLayer(nn.Module):

  def __init__(self,
               obs_space: gym.spaces.Dict,
               embedding_dim: int,
               out_dim: int,
               use_index_select: bool = True,
               activation: Callable = nn.LeakyReLU):
    super(ConvEmbeddingInputLayer, self).__init__()

    embeddings = {}
    n_continuous_channels = 0
    n_embedding_channels = 0
    self.keys_to_op = {}
    for key, val in obs_space.spaces.items():
      assert val.shape == (1, BOARD_SIZE, BOARD_SIZE), f"{key}={val.shape}"
      if isinstance(val, gym.spaces.MultiBinary) or isinstance(
          val, gym.spaces.MultiDiscrete):
        if isinstance(val, gym.spaces.MultiBinary):
          n_embeddings = 2
          padding_idx = 0
        elif isinstance(val, gym.spaces.MultiDiscrete):
          if val.nvec.min() != val.nvec.max():
            raise ValueError(
                f"MultiDiscrete observation spaces must all have the same number of embeddings. "
                f"Found: {np.unique(val.nvec)}")
          n_embeddings = val.nvec.ravel()[0]
          padding_idx = None
        else:
          raise NotImplementedError(f"Got gym space: {type(val)}")
        embeddings[key] = nn.Embedding(n_embeddings,
                                       embedding_dim,
                                       padding_idx=padding_idx)
        n_embedding_channels += embedding_dim
        self.keys_to_op[key] = "embedding"
      elif isinstance(val, gym.spaces.Box):
        n_continuous_channels += 1  # assuming all elements having the same meaning
        self.keys_to_op[key] = "continuous"
      else:
        raise NotImplementedError(
            f"{type(val)} is not an accepted observation space.")

    self.embeddings = nn.ModuleDict(embeddings)
    continuous_space_embedding_layers = []
    embedding_merger_layers = []
    merger_layers = []
    # logging.info(
    # f'n_continuous_channels={n_continuous_channels}, n_embedding_channels={n_embedding_channels}'
    # )

    continuous_space_embedding_layers.extend(
        [nn.Conv2d(n_continuous_channels, out_dim, (1, 1)),
         activation()])
    embedding_merger_layers.extend(
        [nn.Conv2d(n_embedding_channels, out_dim, (1, 1)),
         activation()])
    merger_layers.append(nn.Conv2d(out_dim * 2, out_dim, (1, 1)))

    self.continuous_space_embedding = nn.Sequential(
        *continuous_space_embedding_layers)
    self.embedding_merger = nn.Sequential(*embedding_merger_layers)
    self.merger = nn.Sequential(*merger_layers)
    self.select = _get_select_func(use_index_select)

  def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
    continuous_outs = []
    embedding_outs = {}
    for key, op in self.keys_to_op.items():
      # print(x.keys(), key)
      in_tensor = x[key]
      if op == "embedding":
        # (b, 1, x, y, n_embeddings)
        # drop 1, it's useless
        out = self.select(self.embeddings[key], in_tensor)

        # move channel into second column.
        assert out.shape[1] == 1
        out = out.squeeze(1).permute(0, 3, 1, 2)
        assert len(
            out.shape
        ) == 4, f"Expect embedding to have 5 dims, get {len(out.shape)}: in_shape={in_tensor.shape}{out.shape}"
        embedding_outs[key] = out
        # print('Embedding, ', key, out.shape, in_tensor.shape)
      elif op == "continuous":
        out = in_tensor  #.unsqueeze(-3)
        assert len(in_tensor.shape) == 4, in_tensor.shape
        continuous_outs.append(out)
        # print("contiguous , ", key, out.shape, in_tensor.shape)
      else:
        raise RuntimeError(f"Unknown operation: {op}")

    continuous_out_combined = self.continuous_space_embedding(
        torch.cat(continuous_outs, dim=1))
    embedding_outs_combined = self.embedding_merger(
        torch.cat([v for v in embedding_outs.values()], dim=1))

    # print('continuous_out_combined shape, ', continuous_out_combined.shape)
    # print('embedding_outs_combined shape, ', embedding_outs_combined.shape)
    merged_outs = self.merger(
        torch.cat([continuous_out_combined, embedding_outs_combined], dim=1))
    return merged_outs


class ResidualBlock(nn.Module):

  def __init__(self,
               in_channels: int,
               out_channels: int,
               height: int,
               width: int,
               kernel_size: int = 3,
               normalize: bool = False,
               activation: Callable = nn.ReLU,
               squeeze_excitation: bool = True,
               rescale_se_input: bool = True,
               **conv2d_kwargs):
    super(ResidualBlock, self).__init__()

    # Calculate "same" padding
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # https://www.wolframalpha.com/input/?i=i%3D%28i%2B2x-k-%28k-1%29%28d-1%29%2Fs%29+%2B+1&assumption=%22i%22+-%3E+%22Variable%22
    assert "padding" not in conv2d_kwargs.keys()
    k = kernel_size
    d = conv2d_kwargs.get("dilation", 1)
    s = conv2d_kwargs.get("stride", 1)
    padding = (k - 1) * (d + s - 1) / (2 * s)
    assert padding == int(
        padding), f"padding should be an integer, was {padding:.2f}"
    padding = int(padding)

    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=(kernel_size, kernel_size),
                           padding=(padding, padding),
                           **conv2d_kwargs)

    # We use LayerNorm here since the size of the input "images" may vary based on the board size
    self.norm1 = nn.LayerNorm([in_channels, height, width
                               ]) if normalize else nn.Identity()
    self.act1 = activation()

    self.conv2 = nn.Conv2d(in_channels=out_channels,
                           out_channels=out_channels,
                           kernel_size=(kernel_size, kernel_size),
                           padding=(padding, padding),
                           **conv2d_kwargs)
    self.norm2 = nn.LayerNorm([in_channels, height, width
                               ]) if normalize else nn.Identity()
    self.final_act = activation()

    if in_channels != out_channels:
      self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
    else:
      self.change_n_channels = nn.Identity()

    if squeeze_excitation:
      squeeze_channels = out_channels // 16
      self.squeeze_excitation = SqueezeExcitation(out_channels,
                                                  squeeze_channels)
    else:
      self.squeeze_excitation = nn.Identity()

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
    identity = x
    x = self.conv1(x)
    x = self.act1(self.norm1(x))
    x = self.conv2(x)
    x = self.squeeze_excitation(self.norm2(x))
    x = x + self.change_n_channels(identity)
    return self.final_act(x)


class DictActor(nn.Module):

  def __init__(
      self,
      in_channels: int,
      action_space: gym.spaces.Discrete,
  ):
    super(DictActor, self).__init__()
    self.action_space = action_space
    conv2d = nn.Conv2d(in_channels, in_channels, (1, 1))
    flatten = nn.Flatten(start_dim=1, end_dim=-1)
    linear = nn.Linear(in_channels * BOARD_SIZE * BOARD_SIZE, action_space.n)
    self.actor = nn.Sequential(conv2d, flatten, linear)

  def forward(self, x: torch.Tensor,
              sample: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = self.actor(x)
    actions = DictActor.logits_to_actions(logits, sample)
    return logits, actions

  @staticmethod
  @torch.no_grad()
  def logits_to_actions(logits: torch.Tensor, sample: bool) -> int:
    if sample:
      policy = Categorical(logits=logits)
      return policy.sample()
    else:
      return logits.argsort(dim=-1, descending=True)[:, 1]


class BaselineLayer(nn.Module):

  def __init__(self, in_channels: int, reward_space: RewardSpec):
    super(BaselineLayer, self).__init__()
    self.reward_min = reward_space.reward_min
    self.reward_max = reward_space.reward_max
    self.linear = nn.Linear(in_channels, 1)
    if reward_space.zero_sum:
      self.activation = nn.Softmax(dim=-1)
    else:
      self.activation = nn.Sigmoid()

  def forward(self, x: torch.Tensor):
    # Average feature planes
    x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)

    # Project and reshape input
    x = self.linear(x)
    # Rescale to [0, 1], and then to the desired reward space
    x = self.activation(x)
    return x * (self.reward_max - self.reward_min) + self.reward_min


class BasicActorCriticNetwork(nn.Module):

  def __init__(
      self,
      base_model: nn.Module,
      base_out_channels: int,
      action_space: gym.spaces.Dict,
      reward_space: RewardSpec,
      actor_critic_activation: Callable = nn.ReLU,
      n_action_value_layers: int = 2,
  ):
    super(BasicActorCriticNetwork, self).__init__()
    self.dict_input_layer = DictInputLayer()
    self.base_model = base_model
    self.base_out_channels = base_out_channels

    if n_action_value_layers < 2:
      raise ValueError(
          "n_action_value_layers must be >= 2 in order to use spectral_norm")

    self.actor_base = self.make_spectral_norm_head_base(
        n_layers=n_action_value_layers,
        n_channels=self.base_out_channels,
        activation=actor_critic_activation)
    self.actor = DictActor(self.base_out_channels, action_space)

    self.baseline_base = self.make_spectral_norm_head_base(
        n_layers=n_action_value_layers,
        n_channels=self.base_out_channels,
        activation=actor_critic_activation)
    self.baseline = BaselineLayer(
        in_channels=self.base_out_channels,
        reward_space=reward_space,
    )

  def forward(self,
              x: Dict[str, Union[dict, torch.Tensor]],
              sample: bool = True,
              **actor_kwargs) -> Dict[str, Any]:
    x = self.dict_input_layer(x)
    base_out = self.base_model(x)
    policy_logits, actions = self.actor(self.actor_base(base_out),
                                        sample=sample,
                                        **actor_kwargs)
    baseline = self.baseline(self.baseline_base(base_out))
    return dict(actions=actions,
                policy_logits=policy_logits,
                baseline=baseline)

  def sample_actions(self, *args, **kwargs):
    return self.forward(*args, sample=True, **kwargs)

  def select_best_actions(self, *args, **kwargs):
    return self.forward(*args, sample=False, **kwargs)

  @staticmethod
  def make_spectral_norm_head_base(n_layers: int, n_channels: int,
                                   activation: Callable) -> nn.Module:
    """
        Returns the base of an action or value head, with the final layer of the base/the semifinal layer of the
        head spectral normalized.
        NB: this function actually returns a base with n_layer - 1 layers, leaving the final layer to be filled in
        with the proper action or value output layer.
        """
    assert n_layers >= 2
    layers = []
    for i in range(n_layers - 2):
      layers.append(nn.Conv2d(n_channels, n_channels, (1, 1)))
      layers.append(activation())
    layers.append(
        nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, (1, 1))))
    layers.append(activation())
    return nn.Sequential(*layers)


def create_model(flags, game_env,
                 device: torch.device) -> nn.Module:
  return _create_model(game_env.observation_space,
                       game_env.action_space,
                       embedding_dim=flags.embedding_dim,
                       hidden_dim=flags.hidden_dim,
                       n_blocks=flags.n_blocks,
                       device=device)


def _create_model(observation_space,
                  action_space,
                  embedding_dim=32,
                  hidden_dim=128,
                  n_blocks=4,
                  device: torch.device = torch.device('cpu')):
  base_model = nn.Sequential(
      ConvEmbeddingInputLayer(observation_space, embedding_dim, hidden_dim), *[
          ResidualBlock(in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        height=BOARD_SIZE,
                        width=BOARD_SIZE) for _ in range(n_blocks)
      ])
  model = BasicActorCriticNetwork(base_model, hidden_dim, action_space,
                                  reward_spec)
  return model.to(device=device)
