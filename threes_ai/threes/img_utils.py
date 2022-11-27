import logging
import math

import numpy as np
from matplotlib import pyplot as plt
import cv2

BOARD_X = 137 + 1
BOARD_Y = 240 + 5

CANDIDATE_X = 352 + -3
CANDIDATE_Y = 76 + 1

BLACK_THRESHOLD = 30


def plot_img(img):
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()


class CardExtractor:
  """CardExtractor takes a image of size (1944, 2592), extracts
  the cards on the board and candidate position."""

  def __init__(self, img_path, debug=False):
    self.img_path = img_path
    self.debug = debug

  def rough_crop(self, img):
    h1, h2 = 100, 1800
    w1, w2 = 600, 1900
    return img[h1:h2, w1:w2]

  def detect_ipad_display(self, img):
    CUT_THRESHOLD = 80
    MIN_AREA = 100000
    MAX_AREA = 1000000

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh_img = cv2.threshold(gray, CUT_THRESHOLD, 255,
                                  cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)
    selected_contours = [
        cnt for cnt in contours if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA
    ]
    contour = np.vstack(selected_contours)
    rect = cv2.minAreaRect(contour)
    return rect

  def calibrated_crop(self, img, rect):

    def compute_rtation_angle(rect):
      box_points = cv2.boxPoints(rect)
      points = sorted(box_points.tolist(), key=lambda p: p[1])[:2]
      points = sorted(points, key=lambda p: p[0])
      ((x1, y1), (x2, y2)) = points
      return np.rad2deg(np.arctan2(y2 - y1, x2 - x1))

    center, size = rect[:2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    angle = compute_rtation_angle(rect)

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    logging.debug("width: {}, height: {}, angle(computed)={}".format(
        width, height, angle))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    x, y = size
    if x < y:
      x, y = y, x
    img_crop = cv2.getRectSubPix(img_rot, (y, x), center)
    return img_rot, img_crop

  def resize(self, src):
    scale_percent = src.shape[0] / 1000
    width = int(src.shape[1] / scale_percent)
    height = int(src.shape[0] / scale_percent)
    return cv2.resize(src, (width, height))

  def extract(self):
    img = cv2.imread(self.img_path)
    cropped_img = self.rough_crop(img)
    rect = self.detect_ipad_display(cropped_img)
    _, center_img = self.calibrated_crop(cropped_img, rect)
    center_img = self.resize(center_img)

    candi_cards_extractor = CandidateCardsExtractor(center_img,
                                                    debug=self.debug)
    candidate_card_imgs = candi_cards_extractor.extract()

    board_cards_extractor = BoardCardsExtractor(center_img, debug=self.debug)
    board_cards_imgs = board_cards_extractor.extract()
    return candidate_card_imgs, board_cards_imgs


class CandidateCardsExtractor:
  """Given a display img, extract candidate cards."""

  def __init__(self, img, debug=False):
    self.img = img
    self.debug = debug

  def compute_mid_black_width(self, card_img):
    mid_y = int(card_img.shape[0] * 0.5)
    mid_pixels = card_img[mid_y, :]

    blacks = mid_pixels < BLACK_THRESHOLD
    left_index = np.argmax(blacks)
    right_index = (blacks.cumsum() * blacks).argmax()
    if self.debug:
      yy1 = card_img.shape[0] // 2 - 5
      yy2 = card_img.shape[0] // 2 + 5
      plot_img(card_img[yy1:yy2, :])
    return right_index - left_index

  def compute_num_cards_by_width(self, img):

    def locate_candidate_cards(img):
      h1, h2, w1, w2 = 70, 160, 260, 500
      return img[h1:h2, w1:w2]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    card_img = locate_candidate_cards(img)
    if self.debug:
      plot_img(card_img)

    black_width = self.compute_mid_black_width(card_img)
    num_cards = int(math.ceil(black_width / 45))
    logging.info(
        f"num_cards={num_cards}, black_width={black_width}, candidate_area_width={card_img.shape[1]}"
    )
    return max(min(num_cards, 3), 1)

  def extract(self):
    num_cards = self.compute_num_cards_by_width(self.img)

    CARD_WIDTH = 50
    CARD_HEIGHT = 67
    DELTA_WIDTH = CARD_WIDTH + 23

    def w1_(d=0):
      return CANDIDATE_X + d * DELTA_WIDTH

    h1 = CANDIDATE_Y
    h2 = CARD_HEIGHT + CANDIDATE_Y
    w1 = w1_(-(num_cards - 1) / 2)

    imgs = []
    for i in range(num_cards):
      w2 = w1 + CARD_WIDTH
      c = self.img[int(h1):int(h2), int(w1):int(w2)]
      imgs.append(c)
      w1 += DELTA_WIDTH

      if self.debug:
        logging.info(f'shape={c.shape}')
        plot_img(c)
    return imgs


class BoardCardsExtractor:

  def __init__(self, img, debug=False):
    self.img = img
    self.debug = debug

  def extract(self):
    CARD_WIDTH = 107
    CARD_HEIGHT = 146
    MARGIN_W = 16
    MARGIN_H = 20

    imgs = []
    dw = CARD_WIDTH + MARGIN_W
    dh = CARD_HEIGHT + MARGIN_H
    for i in range(4):
      h1 = BOARD_Y + i * dh
      h2 = h1 + CARD_HEIGHT

      row_imgs = []
      for j in range(4):
        w1 = BOARD_X + j * dw
        w2 = w1 + CARD_WIDTH
        c = self.img[h1:h2, w1:w2]
        row_imgs.append(c)

        if self.debug:
          logging.info(f'({i}, {j}), shape={c.shape}')
          plot_img(c)
      imgs.append(row_imgs)
    return imgs
