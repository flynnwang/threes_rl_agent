"""flask --app tools.camera_app run
"""

import io

from flask import Flask, send_file

app = Flask(__name__)

TEST_IMG_PATH = "/Users/flynn.wang/repo/flynn/thress_imgs/record_1124/002a1a0c-bd7d-4d36-ac1c-fdf5f7d2fdfb.jpg"


def get_test_img_binary():
  with open(TEST_IMG_PATH, 'rb') as f:
    return io.BytesIO(f.read())


@app.route("/board.jpg")
def get_img():
  img_file = get_test_img_binary()
  return send_file(img_file,
                   mimetype='image/jpeg',
                   as_attachment=True,
                   download_name='board.jpg')
