
import os
import random

from playsound import playsound

from threes_ai.threes.consts import MoveDirection

# path = "/Users/flynn.wang/Downloads/向右打.m4a"
# playsound(path)


SOUND_DIR = "/Users/flynn.wang/repo/flynn/thress_imgs/move_sound"


def random_sound(dir: MoveDirection):
  dir = str(dir).split('.')[-1]
  sound_dir = os.path.join(SOUND_DIR, dir)

  sound_files = os.listdir(sound_dir)
  sound_files = [os.path.join(SOUND_DIR, dir, x)
                 for x in sound_files if x.endswith("m4a")]
  sound_file = random.choice(sound_files)
  playsound(sound_file)


# random_sound(MoveDirection.LEFT)
# random_sound(MoveDirection.RIGHT)
# random_sound(MoveDirection.UP)
random_sound(MoveDirection.DOWN)

# playsound("/Users/flynn.wang/repo/flynn/thress_imgs/move_sound/LEFT/an_left.m4a")
