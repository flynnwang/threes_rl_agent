"""flask --app tools.camera_app run
"""

import io

from flask import Flask, send_file

app = Flask(__name__)

TEST_IMG_PATH = "/Users/flynn.wang/repo/flynn/thress_imgs/record_1124/002a1a0c-bd7d-4d36-ac1c-fdf5f7d2fdfb.jpg"

DEBUG = True

if not DEBUG:
  import picamera
  camera = picamera.PiCamera()
  camera.resolution = (2592, 1944)
  # Start a preview and let the camera warm up for 2 seconds
  camera.start_preview()
  time.sleep(2)


def capture_image():
  if DEBUG:
    with open(TEST_IMG_PATH, 'rb') as f:
      return io.BytesIO(f.read())

  img = BytesIO()
  camera.capture(img, 'jpeg')
  return img


@app.route("/board.jpg")
def get_img():
  img = capture_image()
  return send_file(img,
                   mimetype='image/jpeg',
                   as_attachment=True,
                   download_name='board.jpg')
