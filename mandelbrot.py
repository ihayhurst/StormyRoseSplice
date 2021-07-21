#!/usr/bin/env python3
# coding: utf-8
import sys, math, copy, os
import numpy as np
from PIL import Image, ImageOps
from numba import jit, int32, complex64
from functools import cmp_to_key
import pyopencl as cl
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm

@jit
def mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2.0** 40:
            return n
        z = z * z + c
    return 0

def mandelbrot_gpu(q, maxiter):
    queue = cl.CommandQueue(settings.context)
    output = np.empty(q.shape, dtype=np.uint16)

    prg = cl.Program(settings.context, """
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
    q_opencl = cl.Buffer(settings.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(settings.context, mf.WRITE_ONLY, output.nbytes)
    prg.mandelbrot(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxiter))
    cl.enqueue_copy(queue, output, output_opencl).wait()
    return output

#Progress bar to get an idea of when the image will be finished
def progressIndication(x, screenSize):
    if x%32 == 0:
        prog = round(x / screenSize * 100)
        print(str(prog) + "% done", end="\r")

#Used for processing on CPU
def mandelbrot_iterate_bypixel(xmin, xmax, ymin, ymax, width, height, maxiter):
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

#Used for processing on GPU
def mandelbrot_iterate_byarray(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.arange(xmin, xmax, (xmax - xmin) / width)
    r2 = np.arange(ymin, ymax, (ymax - ymin) / height) * 1j
    c = r1 + r2[:,np.newaxis]
    c = np.ravel(c).astype(np.complex64)
    n3 = mandelbrot_gpu(c, maxiter)
    n3 = (n3.reshape((height, width)) / float(n3.max()) * 255.).astype(np.uint8)
    return (n3)

def mandelbrot_iterate(xmin, xmax, ymin, ymax, width, height, maxiter):
    if settings.iterateMethod == 'array':
        return mandelbrot_iterate_byarray(xmin, xmax, ymin, ymax, width, height, maxiter)
    elif settings.iterateMethod == 'pixel':
        return mandelbrot_iterate_bypixel(xmin, xmax, ymin, ymax, width, height, maxiter)
    else:
        print(f'Unrecognised iteration method "{settings.iterateMethod}", exiting')
        exit(1)

def mandelbrot_prepare(coords):
    sideLength = coords['x']['max'] - coords['x']['min']
    if sideLength != (coords['y']['max'] - coords['y']['min']):
        print('X and Y length must be the same')
        exit(1)

    class tile:
        def __init__(self, total):
            self.totalTiles = total
            self.tilesPerSide = int(math.sqrt(self.totalTiles))
            self.length = int(imageSettings.length / self.tilesPerSide)
            self.maxIter = imageSettings.maxIter

        def setXInfo(self, xCount):
            self.x['min'] = coords['x']['min'] + ((sideLength / self.tilesPerSide) * xCount)
            self.x['max'] = coords['x']['min'] + ((sideLength / self.tilesPerSide) * (xCount + 1))

        def setYInfo(self, yCount):
            self.y['min'] = coords['y']['min'] + ((sideLength / self.tilesPerSide) * yCount)
            self.y['max'] = coords['y']['min'] + ((sideLength / self.tilesPerSide) * (yCount + 1))

        x = {}
        y = {}

    tileCount = imageSettings.tilesPerSide ** 2
    tileInfo = tile(tileCount)
    tiles = []

    for x in range(0, tileInfo.tilesPerSide):
        tempTile = tile(tileCount)
        tempTile.setXInfo(x) #Work out x min and max values for that tile
        for y in range(0, tileInfo.tilesPerSide):
            tempTile.id = (tileInfo.tilesPerSide - y) + (x * tileInfo.tilesPerSide) #Get the tile ID, using x and y
            tempTile.setYInfo(y) #Work out y min and max values for that tile

            tiles.append(copy.deepcopy(tempTile))
            tiles[len(tiles) - 1].x = copy.deepcopy(tempTile.x)
            tiles[len(tiles) - 1].y = copy.deepcopy(tempTile.y)

    def orderById(a, b):
        if a.id > b.id:
            return 1
        else:
            return -1
    tiles.sort(key = cmp_to_key(orderById))

    for tile in tiles:
        print(f'{tile.id}: {tile.x["min"]}, {tile.x["max"]}, {tile.y["min"]}, {tile.y["max"]}, {tile.length}, {tile.length}, {tile.maxIter}')
        data = mandelbrot_iterate(tile.x['min'], tile.x['max'], tile.y['min'], tile.y['max'], tile.length, tile.length, tile.maxIter)
        mandelbrot_image_pil(data, tile, f'plots/plot{tile.id}.png')

    #Merge all the images into 1
    combine_images('output')

def mandelbrot_image_mpl(data, tile, outputPath):
    cmap = 'twilight'
    imageDpi = 100

    fig, ax = plt.subplots(figsize = ((tile.length / imageDpi), (tile.length / imageDpi)), dpi = imageDpi)

    ax.axis('off')
    norm = colors.PowerNorm(0.5 * imageSettings.colourMultiplier)
    ax.imshow(data, cmap=cmap, norm=norm, origin='lower')

    if not os.path.isdir('plots'):
        os.mkdir('plots')

    fig.tight_layout()
    fig.savefig(outputPath, bbox_inches='tight', pad_inches=0)
    print(f'Rendered region {tile.id}/{tile.totalTiles}\n')

    plt.clf()
    plt.close()

def mandelbrot_image_pil(data, tile, outputPath):
    #Normalise data
    data = data/(data.max()/1.0)

    #Apply a colourmap, remap to 0-255
    colourMap = cm.get_cmap('twilight')
    data = np.uint8(colourMap(data) * 255)

    #Create image and flip
    image = Image.fromarray(data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(outputPath)
    print(f'Rendered region {tile.id}/{tile.totalTiles}\n')

def combine_images(outputImage):
    outputPath = f'{outputImage}.{imageSettings.fileType}'
    print(f'Exporting images to {outputPath}')

    images = []
    for file in os.listdir('plots'):
        if file.endswith(imageSettings.fileType):
            images.append(f'plots/{file}')

    def plotSort(a, b):
        #Remove 'plots/plot' and file extension from files, then compare numerically
        a = a.split('.')[0].replace('plots/plot', '')
        b = b.split('.')[0].replace('plots/plot', '')
        if int(a) > int(b):
            return 1
        else:
            return -1

    images.sort(key = cmp_to_key(plotSort))

    shape = (imageSettings.tilesPerSide, imageSettings.tilesPerSide)
    length = int(imageSettings.length / imageSettings.tilesPerSide)
    size  = (length, length)

    shape = shape if shape else (1, len(images))
    image_size = (length * shape[1], length * shape[0])
    image = Image.new('RGB', image_size)
    
    #Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = length * col, length * row

            imageCount = col * shape[1] + row
            images[imageCount] = Image.open(images[imageCount])

            image.paste(images[imageCount], offset)
            images[imageCount] = None
    image.save(outputPath, imageSettings.fileType)


def main():
    #Set co-ordinate boundary below
    coords = {
        'default': {
            'x': {
                'min': -0.74877,
                'max': -0.74872
            },
            'y': {
                'min': 0.065053,
                'max': 0.065103
            }
        },
        'alternative': {
            'x': {
                'min': -0.750222,
                'max': -0.749191
            },
            'y': {
                'min': 0.030721,
                'max': 0.031752
            }
        }
    }

    #Cleanup old plots
    oldPlots = os.listdir('plots')
    if oldPlots != []:
        for file in oldPlots:
          os.remove(f'plots/{file}')

    mandelbrot_prepare(coords['default'])

    print('Image complete!')
    sys.exit(1)

#Program operation settings
class settings:
    iterateMethod = 'pixel' #Handled by arguments, 'pixel' uses cpu
    context = '' #Handled by arguments
    coreCount = 1

#Settings to generate image from
class imageSettings:
    def __init__(self):
        self.fileType = 'png'
        self.tilesPerSide = 2
        self.resolutionMultiplier = 1
        self.length = 4096 * self.resolutionMultiplier
        self.maxIter = 2048
        self.colourMultiplier = 1
imageSettings = imageSettings()

class deviceSettings:
    defaultPlatform = 0
    defaultDevice = 0
    #Handled by arguments from here
    platform = None
    device = None
    useSettings = False

def createContext():
    #Don't change context if one is already selected
    if settings.context != '':
        return settings.context

    #Turn pre-selected device into a context
    if deviceSettings.useSettings:
        platform = cl.get_platforms()[deviceSettings.platform] #Select the chosen platform
        device = platform.get_devices()[deviceSettings.device] #Select the chosen device
        return cl.Context([device])

    #Ask for a device and return it
    return cl.create_some_context(interactive=True)

if __name__ == '__main__':
    for i, arg in enumerate(sys.argv):
        if arg == '--cpu': #Setup settings for CPU processing
            settings.iterateMethod = 'pixel'
        elif arg == '--gpu': #Setup settings for GPU processing
            settings.iterateMethod = 'array'
        elif arg == '--platform': #If a platform is specified, get the next argument and set it as the platform
            if len(sys.argv) >= i + 2:
                platform = sys.argv[i + 1]
                try:
                    platform = int(platform)
                    deviceSettings.platform = platform
                    deviceSettings.useSettings = True
                except:
                    print(f'--platform must be one of the following, not "{platform}":')
                    for i, platform in enumerate(cl.get_platforms()):
                        print(f'{i}: {platform}')
                    exit(1)
        elif arg == '--device': #If a device is specified, get the next argument and set it as the device
            if len(sys.argv) >= i + 2:
                device = sys.argv[i + 1]
                try:
                    device = int(device)
                    deviceSettings.device = device
                    deviceSettings.useSettings = True
                except:
                    print(f'--device must be an integer, not "{device}"')
                    exit(1)
        elif arg == '--help':
            print('Help page:')
            print('  --cpu                 : Run the program on CPU')
            print('  --gpu                 : Run the program on GPU')
            print('  --platform [PLATFORM] : Select which platform the device is on')
            print('  --device [DEVICE]     : Select the device to run on')
            exit(0)

    #If creating a context is required, do it
    if settings.iterateMethod == 'array':
        #If settings are to be used, fill in defaults where needed
        if deviceSettings.useSettings:
            if deviceSettings.platform == None:
                deviceSettings.platform = deviceSettings.defaultPlatform
            if deviceSettings.device == None:
                deviceSettings.device = deviceSettings.defaultDevice

        settings.context = createContext()
    main()
