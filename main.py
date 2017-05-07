from colors import bcolors
print bcolors.HEADER+ 'Loading imutils'
import imutils as imutils

print 'Loading opencv'
import cv2
print 'Loading numpy'

import numpy as np
from functions import *


import argparse

print 'Loading snakes'
import morphsnakes

parser = argparse.ArgumentParser(description='Lung segmentation')
parser.add_argument('input', metavar='F', type=str,
                    help='Input file route')

# Parse arguments
args = parser.parse_args()
input_path = args.input

# Load image
im = cv2.imread(input_path)

print bcolors.HEADER+'Loaded image '+input_path + bcolors.ENDC

cv2.imshow('window', im)
print im.shape


imbw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Clean the center of the image


output = imbw * 1


centerx = 351
centery = 263
radius = 50



# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
# imbw = clahe.apply(imbw)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# imbw = cv2.bilateralFilter(imbw, 21, 20000, 0)
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
# imbw = clahe.apply(imbw)
imbw = cv2.filter2D(imbw, -1, kernel)


#BUENOS
# sig = 5.48
# alph = 3000
# smooth = 1
# thres = 0.35
# ball = 1.48
# iter = 230
#
#Buenos de verdad excepto 3 6 y 10
# sig = 5.3
# alph = 3500
# smooth = 1
# thres = 0.35
# ball = 1.46
# iter = 230

sig = 5.3
alph = 2500
smooth = 1
thres = 0.395
ball = 1.6#1.48
iter = 250




dst = cv2.GaussianBlur(imbw,(5,5),0)
dst = imbw
img = dst/255.0

gI = morphsnakes.gborders(img, alpha=alph, sigma=sig)
# cv2.imshow("Cost", gI)

mgac = morphsnakes.MorphGAC(gI, smoothing=smooth, threshold=thres, balloon=ball)
# mgac.levelset = circle_levelset(img.shape, (300,160), 15)
mgac.levelset = circle_levelset(img.shape, (302,175), 15)

# macwe = morphsnakes.MorphACWE(img, smoothing=1, lambda1=1, lambda2=5)
# macwe.levelset = circle_levelset(img.shape, (255, 255), 25)

mask1, edges = morphsnakes.evolve(mgac, num_iters=iter, animate=True, background=dst)
# cv2.waitKey(0)

mgac = morphsnakes.MorphGAC(gI, smoothing=smooth, threshold=thres, balloon=ball)
mgac.levelset = circle_levelset(img.shape, (300,353), 15)

mask2, edges = morphsnakes.evolve(mgac, num_iters=iter, animate=True, background=dst)


# cv2.waitKey(0)

cv2.imshow("results2",cv2.addWeighted(mask1,0.5,mask2,0.5,0))


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(27,27 ))
closing1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)


closing2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
test = cv2.addWeighted(closing1,0.5,closing2,0.5,0)
cv2.imshow("results3 ",cv2.addWeighted(imbw,0.5,test,0.7,0))



im2, contours, hierarchy = cv2.findContours(closing1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
im22, contour2, hierarchy2 = cv2.findContours(closing2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imbw, contours, -1, 255, 1)
cv2.drawContours(imbw, contour2, -1, 255, 1)
cv2.imshow("contours", imbw)

cv2.waitKey()
