import hydra
from omegaconf import OmegaConf, DictConfig

from threes_ai.threes.hanlder import StepHandler


@hydra.main(config_path="conf", config_name="step_handler_config")
def main(flags: DictConfig):
  handler = StepHandler(flags)
  handler.execute(flags.test_img_path)


if __name__ == "__main__":
  main()
