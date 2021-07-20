#!/usr/bin/env python3
# coding: utf-8
import sys
import numpy as np
from PIL import Image
from numba import jit, int32, complex64
import pyopencl as cl
from matplotlib import pyplot as plt
from matplotlib import colors

#Uncomment next line to allow choice of opencl device
ctx = cl.create_some_context(interactive=True)

#Preselect opencl device (or set an env var  PYOPENCL_CTX='0:0' before running)
#platform = cl.get_platforms()[0]  # Select the first platform [0]
#device = platform.get_devices()[0]  # Select the first device on this platform [0]
#ctx = cl.Context([device])

@jit
def mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2.0** 40:
            return n
        z = z * z + c
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
    if x%32 == 0:
        prog = round(x / screenSize * 100)
        print(str(prog) + "% done", end="\r")

def mandel_iterate_bypixel(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.arange(xmin, xmax, (xmax - xmin) / width)
    r2 = np.arange(ymin, ymax, (ymax - ymin) / height) 
    data = np.zeros((height, width, 3), dtype = np.uint8)
    for x in range(0, height - 1):
        for y in range(0, width - 1):
            c = r1[y] + r2[x] * 1j
            delta = mandelbrot(c, maxiter) 
            red   = int(delta % 255)
            green = int(delta % 128)
            blue  = int(delta % 64)

            data[x, y] = (red, green, blue)
        progressIndication(x, height)
    return(data)

def mandel_iterate_byarray(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.arange(xmin, xmax, (xmax - xmin) / width)
    r2 = np.arange(ymin, ymax, (ymax - ymin) / height) * 1j
    c = r1 + r2[:,np.newaxis]
    c = np.ravel(c).astype(np.complex64)
    n3 = mandelbrot_gpu(c, maxiter)
    n3 = (n3.reshape((height, width)) / float(n3.max()) * 255.).astype(np.uint8)
    return (n3)


def mandelbrot_image_mpl(xmin, xmax, ymin, ymax, width, height, maxiter):
    #Displays with matplotlib
    cmap = 'twilight'
    my_dpi = 100
    print(xmin, xmax, ymin, ymax, width, height, maxiter)

    plot_title = cmap, xmin, xmax, ymin, ymax, width, height, maxiter

    #z = mandel_iterate_bypixel(xmin, xmax, ymin, ymax, width, height, maxiter)
    z = mandel_iterate_byarray(xmin, xmax, ymin, ymax, width, height, maxiter)

    fig, ax = plt.subplots(figsize = (width / my_dpi, height / my_dpi), dpi = my_dpi)
    ticks = np.arange(0,width,512)
    plt.gcf().autofmt_xdate()

    x_ticks = xmin + (xmax - xmin) * ticks / width
    y_ticks = ymin + (ymax - ymin) * ticks / width
    plt.xticks(ticks, x_ticks)
    plt.yticks(ticks, y_ticks)

    ax.set_title(plot_title)
    norm = colors.PowerNorm(0.5)
    ax.imshow(z, cmap=cmap, norm=norm, origin='lower') 
    fig.savefig('plot.png')
    print('Created plot using matplotlib\n')
    plt.clf()

def mandelbrot_image_PIL(xmin, xmax, ymin, ymax, width, height, maxiter):
    #Displays with PIL (mono needs palette)
    print(xmin, xmax, ymin, ymax, width, height, maxiter)
    #z = mandel_iterate_bypixel(xmin, xmax, ymin, ymax, width, height, maxiter)
    z = mandel_iterate_byarray(xmin, xmax, ymin, ymax, width, height, maxiter)
    image = Image.fromarray(z)
    #image.putpalette("P",[255, 0, 0, 254, 0, 0, 253, 0, 0, 244, 0, 0])
    image.save("plot.png")
    print('Created plot using PIL\n') 

def main(args=None):
    #Set required resolution, iteration depth and co-ordinate boundary below
    resolutionMultiplier = 2
    width = 2048 * resolutionMultiplier
    height = 2048 * resolutionMultiplier
    maxiter = 2048
    xmin, xmax, ymin, ymax = -0.74877, -0.74872, 0.065053, 0.065103

    mandelbrot_image_mpl(xmin, xmax, ymin, ymax, width, height, maxiter)
    #mandelbrot_image_PIL(xmin, xmax ,ymin, ymax, width, height, maxiter)

    print("Image complete!")
    sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except ValueError:
        print("Give me something to do")
        sys.exit(1)
