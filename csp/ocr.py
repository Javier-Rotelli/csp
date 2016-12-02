import cv2
import numpy as np
from PIL import Image
from tesserwrap import Tesseract

from crop_morphology import process_image

kernel = np.zeros((3, 3), np.uint8)


def ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # th3 = cv2.adaptiveThreshold(res, i, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)
    ret, th3 = cv2.threshold(res, 99, 255, cv2.THRESH_BINARY)

    rows, cols = th3.shape
    resultado = th3[0:rows, 397:493]
    tiempo = th3[0:rows, 0:112]

    resultado = cv2.bitwise_not(resultado)
    resultado = cv2.erode(resultado, kernel, iterations=2)

    tiempo = cv2.bitwise_not(tiempo)
    tiempo = cv2.erode(tiempo, kernel)

    cv2.imshow('inicial', th3)
    th3 = cv2.dilate(th3, kernel, iterations=2)
    cv2.imshow('dilated', th3)
    th3[0:rows, 0:112] = tiempo
    th3[0:rows, 397:493] = resultado

    cv2.imshow('resultado', resultado)
    cv2.imshow('tiempo', tiempo)

    opening = cv2.morphologyEx(th3[0:rows, 112:cols], cv2.MORPH_OPEN, kernel)

    th3[0:rows, 112:cols] = opening
    cv2.imshow('final', th3)

    cv2.imshow('open', opening)
    return th3


def nothing(x):
    pass

if __name__ == '__main__':
    img = cv2.imread('frame40.jpg')
    tr = Tesseract(datadir='/usr/local/share')
    tr.set_page_seg_mode(mode=7)
    tr.set_variable('load_system_dawg', 'false')
    tr.set_variable('load_freq_dawg', 'false')

    while 1:

        binary = ocr(img)
        print(tr.ocr_image(Image.fromarray(binary)))


        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
