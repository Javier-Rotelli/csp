import numpy as np
import cv2
from tesserwrap import Tesseract
from PIL import Image

import ocr

cap = cv2.VideoCapture('test-file-trimmed.mp4')
tr = Tesseract(datadir='/usr/local/share')
tr.set_page_seg_mode(mode=7)
tr.set_variable('load_system_dawg', 'false')
tr.set_variable('load_freq_dawg', 'false')

count = 0
while cap.isOpened():
    ret, frame = cap.read()

    cropped = frame[51:68, 90:410]
    cv2.imshow('cropped', cropped)
    binary = ocr.ocr(cropped)
    if count % 5 == 0:
        print(tr.ocr_image(Image.fromarray(binary)))
        # cv2.imwrite("frame%d.jpg" % count, cropped)  # save frame as JPEG file
    count += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
