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

myconfig = r"--psm 6 --oem 3"

def load_image(file_name):
        img = Image.open(file_name)
        return img

def get_text(file_name):
        img = Image.open(file_name)
        text = tess.image_to_string(img, config=myconfig)
        return text

# def grayscale_img(img):
#         return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def remove_noise(img):
#         return cv2.medianBlur(img, 5)

# def thresholding(img):
#         return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

Image.show(load_image('noise.png'))

print(get_text('noise.png'))