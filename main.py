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





# Find the contour and the surface of the artery using active contours

img = imbw/255.0
gI = morphsnakes.gborders(img, alpha=3000, sigma=5.48)
mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.31, balloon=1.1)
mgac.levelset = circle_levelset(img.shape, (256,128), 10)

mask1, edges = morphsnakes.evolve(mgac, num_iters=250, animate=True, background=imbw)
cv2.waitKey(0)

gI = morphsnakes.gborders(img, alpha=3000, sigma=5.48)
mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.31, balloon=1.2)
mgac.levelset = circle_levelset(img.shape, (256,384), 10)

mask2, edges = morphsnakes.evolve(mgac, num_iters=250, animate=True, background=imbw)


cv2.waitKey(0)

cv2.imshow("results",cv2.addWeighted(mask1,0.5,mask2,0.5,0))

cv2.waitKey()