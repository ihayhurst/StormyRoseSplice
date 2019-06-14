#!/usr/bin/env python3
# coding: utf-8

import numpy
from PIL import Image
from numba import jit

resolutionMultiplier = 2
width = 1920 * resolutionMultiplier
height = 1080 * resolutionMultiplier

zoomControl = 0.8
zoom = 1000*zoomControl*resolutionMultiplier
#Offset to center the image
offset = -width*1.2/2-(height/2)*1j
#Increase iterations to impove the quality of the image
MAX_ITER = 64

#Set up the array for the pixels
data = numpy.zeros((height, width, 3), dtype=numpy.uint8)

@jit
def mandelbrot(c):
    z = c
    for n in range(MAX_ITER):
        if abs(z) > 2.0** 40:
            return n
        z = z*z + c
    return 0

#Progress bar to get an idea of when the image will be finished
def progressIndication(x, screenSize):
    if x%32==0:
        prog = round(x/screenSize*100)
        print(str(prog) + "% done", end="\r")

#Iterate though each pixel and treat them as an imaginary number
for x in range(0, height-1):
    for y in range(0, width-1):
        i = 1j
        c = y + x*i
        c = c + offset
        c = c/zoom
        
        #Check how fast the point is shooting off to infnity
#        delta = abs(mandelbrot( c, iterations ) - mandelbrot( c, iterations+1 ))
        delta = mandelbrot(c) 
        red   = int(delta * 255 /MAX_ITER)
        green =0 #green = int(delta *255 / MAX_ITER)
        blue = 0 #blue  = int(delta *255 / MAX_ITER)
        #if( delta != delta or delta > 255 ):
        #    delta = 255

        data[x, y] = [red, green, blue]
    progressIndication(x, height)

#Save and show the file
print("Image complete!")
image = Image.fromarray(data)
image.save("plot.tiff")
