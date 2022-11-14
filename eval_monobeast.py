import logging
import os
from types import SimpleNamespace
import time
import threading
import traceback
import timeit
import pprint
from multiprocessing import Manager


import hydra
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import torch
from torch import multiprocessing as mp
import wandb

from threes_ai.torchbeast.core import prof
from threes_ai.utils import flags_to_namespace
from threes_ai.model import create_model
from threes_ai.thress_gym import create_env, create_game_env



os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
            "%(message)s"),
    level=0,
)


def do_evaluation(flags, total_games_played, stats, infos):
  for i in infos:
    max_card_items = i['max_card']
    for max_card in max_card_items:
      max_card = str(max_card)
      max_card_count = stats.get(max_card, 0) + 1
      total_games_played += 1

      stats[max_card] = max_card_count
      stats[f'card_{max_card}_pct'] = max_card_count / total_games_played
  return stats, total_games_played


@torch.no_grad()
def act(
    flags: SimpleNamespace,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    actor_model: torch.nn.Module,
    buffers: list,
):
  if flags.debug:
    catch_me = AssertionError
  else:
    catch_me = Exception
  try:
    logging.info("Actor %i started.", actor_index)
    timings = prof.Timings()

    env = create_env(flags, device=flags.actor_device)
    if flags.seed is not None:
      env.seed(flags.seed + actor_index * flags.n_actor_envs)
    else:
      env.seed()

    env_output = env.reset(force=True)
    agent_output = actor_model(env_output)
    while True:
      index = free_queue.get()
      if index is None:
        break

      # Clear buffer
      # buffers[index] = []
      result = []

      # Do new rollout.
      for t in range(flags.unroll_length):
        timings.reset()

        # Do not do sample during evaluation
        agent_output = actor_model(env_output, sample=False)
        timings.time("model")

        env_output = env.step(agent_output["actions"])
        timings.time("step")

        if env_output["done"].any():
          done = env_output["done"]
          max_card = env_output["info"]["max_card"][done == True]
          card_sum = env_output["info"]["card_sum"][done == True]
          game_step_count = env_output["info"]["game_step_count"][done == True]


          result.append({
              'max_card': max_card.tolist(),
              'card_sum': card_sum.tolist(),
              'game_step_count': game_step_count.tolist()
          })
          # print('buffers[index]: ', buffers[index])
          # print(done)
          # print(max_card)
          # print('after - buffers[index]: ', buffers[index])

          env_output = env.reset()

      full_queue.put(result)

  except KeyboardInterrupt:
    pass  # Return silently.
  except catch_me as e:
    logging.error("Exception in worker process %i", actor_index)
    traceback.print_exc()
    print()
    raise e


def evaluate(flags):
  # Necessary for multithreading and multiprocessing
  os.environ["OMP_NUM_THREADS"] = "1"

  manager = Manager()
  # buffers = manager.list()
  # for _ in range(flags.num_buffers):
    # buffers.append(None)


  assert flags.load_dir
  logging.info("Loading checkpoint state...")
  checkpoint_state = torch.load(Path(flags.load_dir) / flags.checkpoint_file,
                                map_location=torch.device("cpu"))
  print(Path(flags.load_dir) / flags.checkpoint_file)
  assert checkpoint_state

  t = flags.unroll_length
  b = flags.batch_size


  game_env = create_game_env()
  actor_model = create_model(flags, game_env, flags.actor_device)
  if checkpoint_state is not None:
    logging.info("Loading model parameters from checkpoint state...")
    actor_model.load_state_dict(checkpoint_state["model_state_dict"])

  actor_model.eval()
  actor_model.share_memory()
  n_trainable_params = sum(p.numel() for p in actor_model.parameters()
                           if p.requires_grad)
  logging.info(f'Evaluating model with {n_trainable_params:,d} parameters.')

  actor_processes = []
  free_queue = mp.SimpleQueue()
  full_queue = mp.SimpleQueue()

  for actor_id in range(flags.num_actors):
    actor_start = threading.Thread if flags.debug else mp.Process
    actor = actor_start(
        target=act,
        args=(
            flags,
            actor_id,
            free_queue,
            full_queue,
            actor_model,
            None,
        ),
    )
    actor.start()
    actor_processes.append(actor)
    time.sleep(0.5)

  step, total_games_played, stats = 0, 0, {}

  def run_evaluation(lock=threading.Lock()):
    """Thread target for the learning process."""
    nonlocal step, total_games_played, stats
    while step < flags.total_steps:
      infos = full_queue.get()
      try:
        stats, total_games_played = do_evaluation(
            flags=flags,
            total_games_played=total_games_played,
            stats=stats,
            infos=infos,
        )
      except Exception as e:
        __import__('ipdb').set_trace()
      with lock:
        step += t * b
        print(step, sorted(list(stats.items())), total_games_played)
        if not flags.disable_wandb:
          wandb.log(stats, step=step)
      free_queue.put(step)

  for m in range(flags.num_buffers):
    free_queue.put(m)

  learner_threads = []
  thread = threading.Thread(target=run_evaluation, name=f"run_evaluation-")
  thread.start()
  learner_threads.append(thread)

  timer = timeit.default_timer
  try:
    while step < flags.total_steps:
      start_step = step
      start_time = timer()

      sleep_time = 10
      time.sleep(sleep_time)

      sps = (step - start_step) / (timer() - start_time)
      bps = (step - start_step) / (t * b) / (timer() - start_time)
      # logging.info(
          # f"Steps {step:d} @ {sps:.1f} SPS / {bps:.1f} BPS. Stats:\n{pprint.pformat(stats)}"
      # )
  except KeyboardInterrupt:
    # Try checkpointing and joining actors then quit.
    return
  else:
    for thread in learner_threads:
      thread.join()
    logging.info(f"Learning finished after {step:d} steps.")
  finally:
    for _ in range(flags.num_actors):
      free_queue.put(1)
    for actor in actor_processes:
      actor.join(timeout=1)


def get_default_flags(flags: DictConfig) -> DictConfig:
  flags = OmegaConf.to_container(flags)

  # Env params
  flags.setdefault("seed", None)

  # Training params
  flags.setdefault("use_mixed_precision", True)

  # Model params
  flags.setdefault("use_index_select", True)
  if flags.get("use_index_select"):
    logging.info(
        "index_select disables padding_index and is equivalent to using a learnable pad embedding."
    )

  # Reloading previous run params
  flags.setdefault("load_dir", None)
  flags.setdefault("checkpoint_file", None)
  flags.setdefault("weights_only", False)
  flags.setdefault("n_value_warmup_batches", 0)

  # Miscellaneous params
  flags.setdefault("disable_wandb", True)
  flags.setdefault("debug", False)
  return OmegaConf.create(flags)


@hydra.main(config_path="conf", config_name="eval_autodl_1080ti_config")
def main(flags: DictConfig):
  cli_conf = OmegaConf.from_cli()
  if Path("config.yaml").exists():
    new_flags = OmegaConf.load("config.yaml")
    flags = OmegaConf.merge(new_flags, cli_conf)

  if flags.get("load_dir", None) and not flags.get("weights_only", False):
    # this ignores the local config.yaml and replaces it completely with saved one
    # however, you can override parameters from the cli still
    # this is useful e.g. if you did total_steps=N before and want to increase it
    logging.info(
        "Loading existing configuration, we're continuing a previous run")
    new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
    # Overwrite some parameters

    flags = OmegaConf.merge(new_flags, flags, cli_conf)

  flags = get_default_flags(flags)
  logging.info(OmegaConf.to_yaml(flags, resolve=True))
  OmegaConf.save(flags, "config_eval.yaml")

  logging.info("disable_wandb? = %s", flags.disable_wandb)
  if not flags.disable_wandb:
    wandb.init(
        config=vars(flags),
        project=flags.project,
        entity=flags.entity,
        group=flags.group,
        name=flags.name,
    )

  flags = flags_to_namespace(OmegaConf.to_container(flags))
  mp.set_sharing_strategy(flags.sharing_strategy)
  evaluate(flags)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  main()
