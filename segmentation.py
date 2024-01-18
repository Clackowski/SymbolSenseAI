import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Users\byrne\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
from PIL import Image
import cv2
import numpy as np

'''Page segmentation modes: 
O Orientation and script detection (OSD) only
1 Automatic page segmentation with OSD. ‘
2 Automatic page segmentation, but no OSD, or OCR.
3 Fully automatic page segmentation, but no OSD. (Default)
4 Assume a single column of text of variable sizes.
5 Assume a single uniform block of vertically aligned text.
6 Assume a single uniform block of textJ
7 Treat the image as a single text line.
8 Treat the image as a single word.
9 Treat the image as a single word in a circle.
10 Treat the image as a single character.
11 Sparse text. Find as much text as possible in no particular order.
12 Sparse text with OSD.
13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract—specific.
'''

def extract_chars(file = '0123456789.png'):
    img = cv2.imread(file)
    h, w, _ = img.shape

    myconfig = r"--psm 7 --oem 3"
    boxes = tess.image_to_boxes(img, config= myconfig)

    #display boxes on original image
    img_copy = img.copy()
    for b in boxes.splitlines():
        b = b.split(' ')
        img_copy = cv2.rectangle(img_copy, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    cv2.imshow('Image',img_copy)
    cv2.waitKey(0)
    # print(boxes)

    num = 0
    for b in boxes.splitlines():
        b = b.split(' ')
        x1, y1, x2, y2 = (int(b[1]), int(b[2]), int(b[3]), int(b[4]))
        img2 = cv2.imread(file)
        crop = img2[y1:y2, x1:x2]
        # cv2.imshow('Image', crop)
        # cv2.waitKey(0)
        cv2.saveImage(f'{num}.png', crop)
        num += 1
