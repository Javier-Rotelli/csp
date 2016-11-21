import numpy as np
import cv2
import pytesseract
from PIL import Image
from crop_morphology import process_image


cap = cv2.VideoCapture('test-file.mp4')

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    cropped = frame[45:70, 90:425]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    cropped_text = process_image(gray)
    # th3 = cv2.adaptiveThreshold(cropped_text, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('frame', cropped_text)
    if count % 240 == 0:
        print(pytesseract.image_to_string(Image.fromarray(cropped_text), config="-psm 7 -load_system_dawg 0 -load_freq_dawg 0"))
        # cv2.imwrite("frame%d.jpg" % count, cropped_text)  # save frame as JPEG file
    count += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
