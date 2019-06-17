#!/usr/bin/env python3
# coding: utf-8
import sys
import numpy as np
from PIL import Image
from numba import jit, int32, complex64
import pyopencl as cl
from matplotlib import pyplot as plt
from matplotlib import colors

#allow choice of opencl device
#ctx = cl.create_some_context(interactive=True)

#Preselect opencl device (or set an env var  PYOPENCL_CTX='0:0' before running)
platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
ctx = cl.Context([device])

@jit
def mandelbrot(c,maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2.0** 40:
            return n
        z = z*z + c
    return 0

def mandelbrot_gpu(q, maxiter):

    global ctx

    queue = cl.CommandQueue(ctx)

    output = np.empty(q.shape, dtype=np.uint16)
    prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float real = q[gid].x;
        float imag = q[gid].y;
        output[gid] = 0;
        for(int curiter = 0; curiter < maxiter; curiter++) {
            float real2 = real*real, imag2 = imag*imag;
            if (real*real + imag*imag > 4.0f){
                 output[gid] = curiter;
                 return;
            }
            imag = 2* real*imag + q[gid].y;
            real = real2 - imag2 + q[gid].x;

        }
    }
    """).build()

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)


    prg.mandelbrot(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output

#Progress bar to get an idea of when the image will be finished
def progressIndication(x, screenSize):
    if x%32==0:
        prog = round(x/screenSize*100)
        print(str(prog) + "% done", end="\r")

@jit
def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i] + 1j*r2[j],maxiter)
    return (r1,r2,n3)

def mandelbrot_set1(xmin,xmax,ymin,ymax,width,height,maxiter):
    data = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(0, height-1):
        for y in range(0, width-1):
            i = 1j
            c = y + x*i
            #c = c + offset
            #c = c/zoom
            #delta = mandelbrot(c) 
            delta = mandelbrot_gpu(c,maxiter) 
            red   = int(delta * 255 /maxiter)
            green =0 #green = int(delta *255 / MAX_ITER)
            blue = 0 #blue  = int(delta *255 / MAX_ITER)

            data[x, y] = [red, green, blue]
        progressIndication(x, height)
    return(data)

def mandelbrot_set3(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    c = np.ravel(c)
    n3 = mandelbrot_gpu(c,maxiter)
    n3 = n3.reshape((width,height))
    return (r1,r2,n3)


def mandelbrot_image(xmin,xmax,ymin,ymax,width,height,maxiter):
    cmap = 'hot'
    print (xmin,xmax,ymin,ymax,width,height,maxiter)
    x,y,z = mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter)
    fig, ax = plt.subplots(figsize=(width, height))
    ticks = np.arange(0,width,1000)
    #x_ticks = xmin + (xmax-xmin)*ticks/width
    #plt.xticks(ticks, x_ticks)
    #y_ticks = ymin + (ymax-ymin)*ticks/width
    #plt.yticks(ticks, y_ticks)
    ax.set_title(cmap)
    ax.imshow(z.T,cmap=cmap,origin='lower') 
    fig.savefig('plot.png')
    print('Created plot\n')
    plt.clf()

def mandelbrot_image1(xmin,xmax,ymin,ymax,width,height,maxiter):
    print (xmin,xmax,ymin,ymax,width,height,maxiter)
    x,y,z = mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter)
    image = Image.fromarray(z.T)
    image.save("plot.tiff")
    return

def main(args=None):


    resolutionMultiplier = 1
    width = 1920 * resolutionMultiplier
    height = 1080 * resolutionMultiplier

    #zoomControl = 0.8
    #zoom = 1000*zoomControl*resolutionMultiplier
    #Offset to center the image
    #offset = -width*1.2/2-(height/2)*1j
    #Increase iterations to impove the quality of the image
    maxiter = 2048
    #make the (xmin,xmax,ymin,ymax,width,height,maxiter):
    mandelbrot_image(-2.0,0.5,-1.25,1.25,width,height,maxiter)
    print("Image complete!")
    sys.exit(1)

if __name__ == '__main__':
    try:
        #main(sys.argv[1:])
        main()
    except ValueError:
        print("Give me something to do")
        sys.exit(1)
