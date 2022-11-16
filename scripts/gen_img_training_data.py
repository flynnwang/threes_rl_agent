import argparse
import os

import cv2
from tqdm import tqdm

from threes_ai.threes.img_utils import CardExtractor, plot_img


def gen_imgs(input_path, output_path, candidate_only=False):

  candi_out_path = os.path.join(output_path, 'candi_imgs')
  if not os.path.exists(candi_out_path):
    os.makedirs(candi_out_path)

  board_out_path = os.path.join(output_path, 'board_imgs')
  if not os.path.exists(board_out_path):
    os.makedirs(board_out_path)

  for f in tqdm(os.listdir(input_path)):
    if not f.endswith('jpg'):
      continue

    uuid = f.split('.')[0]
    img_path = os.path.join(input_path, f)
    ce = CardExtractor(img_path)

    candi_imgs, board_imgs = ce.extract()
    for i, img in enumerate(candi_imgs):
      out_path = os.path.join(candi_out_path, f"{uuid}_{i}.jpg")
      cv2.imwrite(out_path, img)

    if candidate_only:
      continue

    for i in range(4):
      for j in range(4):
        img = board_imgs[i][j]
        out_path = os.path.join(board_out_path, f"{uuid}_{i}_{j}.jpg")
        cv2.imwrite(out_path, img)


def main():
  parser = argparse.ArgumentParser(
      "gen_img_training_data",
      description='Extract data from camera image for training.')
  parser.add_argument('-i', '--input_path', required=True)
  parser.add_argument('-o', '--output_path', required=True)
  parser.add_argument('--candidate-only', action='store_true')
  args = parser.parse_args()

  gen_imgs(args.input_path, args.output_path, args.candidate_only)


if __name__ == "__main__":
  main()
