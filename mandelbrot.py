#!/usr/bin/env python3
# coding: utf-8
import sys, math, copy, os
import numpy as np
from PIL import Image, ImageOps
from numba import jit, int32, complex64
from functools import cmp_to_key
import pyopencl as cl
from matplotlib import colormaps


@jit(nopython=True)
def mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2.0**40:
            return n
        z = z * z + c
    return 0


def mandelbrot_gpu(q, maxiter):
    queue = cl.CommandQueue(settings.context)
    output = np.empty(q.shape, dtype=np.uint16)

    prg = cl.Program(
        settings.context,
        """
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
    """,
    ).build()

    mf = cl.mem_flags
    q_opencl = cl.Buffer(settings.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(settings.context, mf.WRITE_ONLY, output.nbytes)
    prg.mandelbrot(
        queue, output.shape, None, q_opencl, output_opencl, np.uint16(maxiter)
    )
    cl.enqueue_copy(queue, output, output_opencl).wait()
    return output


# Used for processing on CPU
def mandelbrot_iterate(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    c = r1 + r2[:, None] * 1j  # Create a 2D grid of complex numbers

    if settings.calcMethod =="gpu":
        c = np.ravel(c).astype(np.complex64)
        n3 = mandelbrot_gpu(c, maxiter)
    else:
        mandelbrot_vectorized = np.vectorize(mandelbrot)
        n3 = mandelbrot_vectorized(c, maxiter)

    n3 = (n3.reshape((height, width)) / float(n3.max()) * 255.0).astype(np.uint8)
    return n3





def mandelbrot_prepare(coords):
    sideLength = coords["x"]["max"] - coords["x"]["min"]
    if sideLength != (coords["y"]["max"] - coords["y"]["min"]):
        print("X and Y length must be the same")
        exit(1)

    class tile:
        def __init__(self, total):
            self.totalTiles = total
            self.tilesPerSide = int(math.sqrt(self.totalTiles))
            self.length = int(imageSettings.length / self.tilesPerSide)
            self.maxIter = imageSettings.maxIter

        def setXInfo(self, xCount):
            self.x["min"] = coords["x"]["min"] + (
                (sideLength / self.tilesPerSide) * xCount
            )
            self.x["max"] = coords["x"]["min"] + (
                (sideLength / self.tilesPerSide) * (xCount + 1)
            )

        def setYInfo(self, yCount):
            self.y["min"] = coords["y"]["min"] + (
                (sideLength / self.tilesPerSide) * yCount
            )
            self.y["max"] = coords["y"]["min"] + (
                (sideLength / self.tilesPerSide) * (yCount + 1)
            )

        x = {}
        y = {}

    tileCount = imageSettings.tilesPerSide**2
    tileInfo = tile(tileCount)
    tiles = []

    for x in range(0, tileInfo.tilesPerSide):
        tempTile = tile(tileCount)
        tempTile.setXInfo(x)  # Work out x min and max values for that tile
        for y in range(0, tileInfo.tilesPerSide):
            tempTile.id = (tileInfo.tilesPerSide - y) + (
                x * tileInfo.tilesPerSide
            )  # Get the tile ID, using x and y
            tempTile.setYInfo(y)  # Work out y min and max values for that tile

            tiles.append(copy.deepcopy(tempTile))
            tiles[len(tiles) - 1].x = copy.deepcopy(tempTile.x)
            tiles[len(tiles) - 1].y = copy.deepcopy(tempTile.y)

    def orderById(a, b):
        if a.id > b.id:
            return 1
        else:
            return -1

    tiles.sort(key=cmp_to_key(orderById))

    for tile in tiles:
        print(
            f'{tile.id}: {tile.x["min"]}, {tile.x["max"]}, {tile.y["min"]}, {tile.y["max"]}, {tile.length}, {tile.length}, {tile.maxIter}'
        )
        data = mandelbrot_iterate(
            tile.x["min"],
            tile.x["max"],
            tile.y["min"],
            tile.y["max"],
            tile.length,
            tile.length,
            tile.maxIter,
        )
        mandelbrot_image_pil(
            data, tile, f"plots/plot{tile.id}.{imageSettings.fileType}"
        )

    # Merge all the images into 1
    combine_images("output")


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype("float")
    # Do not touch the alpha channel
    for i in range(3):
        # minval = arr[...,i].min()
        minval = 0
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= 255.0 / (maxval - minval)
    return arr


def mandelbrot_image_pil(data, tile, outputPath):
    # Normalise data
    # data = data / (data.max() / 1.0)
    data = normalize(data).astype("uint8")
    # Apply a colourmap, remap to 0-255
    colourMap = colormaps.get_cmap(imageSettings.colourMap)
    data = np.uint8(colourMap(data) * 255)

    # Create image and flip
    # image = Image.fromarray(data)
    image = Image.fromarray(data, "RGBA")
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    image.save(outputPath)
    print(f"Rendered region {tile.id}/{tile.totalTiles}\n")


def combine_images(outputImage):
    outputPath = f"{outputImage}.{imageSettings.fileType}"
    print(f"Exporting images to {outputPath}")

    images = []
    for file in os.listdir("plots"):
        if file.endswith(imageSettings.fileType):
            images.append(f"plots/{file}")

    def plotSort(a, b):
        # Remove 'plots/plot' and file extension from files, then compare numerically
        a = a.split(".")[0].replace("plots/plot", "")
        b = b.split(".")[0].replace("plots/plot", "")
        if int(a) > int(b):
            return 1
        else:
            return -1

    images.sort(key=cmp_to_key(plotSort))

    shape = (imageSettings.tilesPerSide, imageSettings.tilesPerSide)
    length = int(imageSettings.length / imageSettings.tilesPerSide)
    size = (length, length)

    shape = shape if shape else (1, len(images))
    image_size = (length * shape[1], length * shape[0])
    image = Image.new("RGB", image_size)

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = length * col, length * row

            imageCount = col * shape[1] + row
            images[imageCount] = Image.open(images[imageCount])

            image.paste(images[imageCount], offset)
            images[imageCount] = None
    image.save(outputPath, imageSettings.fileType)


"""
  #Top level
    #xmin, xmax, ymin, ymax = -2.0, 0.5,-1.25, 1.25
    #Test Zoom 1
    #xmin,xmax,ymin,ymax = -0.375,0.125, -1.125,-0.675
    #Test Zoom 2
    #min,xmax,ymin,ymax = -0.25,-0.125, -0.9,-0.775
    #Test Zoom 3
    #xmin,xmax,ymin,ymax = -0.1875,-0.1725, -0.845,-0.830
    #Interesting region from JFP's page
    #xmin,xmax,ymin,ymax = -0.74877, -0.74872, 0.065053 , 0.065103
    #dcutchen post marksmath,org
    xmin,xmax,ymin,ymax =-0.34853774148008254, -0.34831493420245574, -0.6065922085831237, -0.606486596104741
    mandelbrot_image_mpl(xmin,xmax,ymin,ymax,width,height,maxiter)
    #mandelbrot_image_PIL(xmin,xmax,ymin,ymax,width,height,maxiter)
"""


def main():
    # Set co-ordinate boundary below
    coords = {
        "default": {
            "x": {"min": -0.74872, "max": -0.74772},
            "y": {"min": 0.06453, "max": 0.06553},
        },
        "alternative": {
            "x": {"min": -0.750222, "max": -0.749191},
            "y": {"min": 0.030721, "max": 0.031752},
        },
    }

    # Cleanup old plots
    if os.path.isdir("plots"):
        oldPlots = os.listdir("plots")
        if oldPlots != []:
            for file in oldPlots:
                os.remove(f"plots/{file}")

    mandelbrot_prepare(coords["default"])

    print("Image complete!")
    sys.exit(1)


# Program operation settings
class settings:
    calcMethod = "cpu"  # Handled by arguments uses cpu
    context = ""  # Handled by arguments
    coreCount = 8


# Settings to generate image from
class imageSettings:
    def __init__(self):
        self.fileType = "png"
        self.tilesPerSide = 2
        self.resolutionMultiplier = 1
        self.length = 4096 * self.resolutionMultiplier
        self.maxIter = 2048
        self.colourMultiplier = 1
        self.colourMap = "magma"


imageSettings = imageSettings()
imageSettings.maxIter=1024


class deviceSettings:
    defaultPlatform = 0
    defaultDevice = 0
    # Handled by arguments from here
    platform = None
    device = None
    useSettings = False


def createContext():
    # Don't change context if one is already selected
    if settings.context != "":
        return settings.context

    # Turn pre-selected device into a context
    if deviceSettings.useSettings:
        platform = cl.get_platforms()[
            deviceSettings.platform
        ]  # Select the chosen platform
        device = platform.get_devices()[
            deviceSettings.device
        ]  # Select the chosen device
        return cl.Context([device])

    # Ask for a device and return it
    return cl.create_some_context(interactive=True)


if __name__ == "__main__":
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        match arg:
            case "--cpu":
                # Setup settings for CPU processing
                settings.calcMethod = "cpu"
            case "--gpu":
                # Setup settings for GPU processing
                settings.calcMethod = "gpu"
            case "--platform":
                # If a platform is specified, get the next argument and set it as the platform
                if i + 1 < len(sys.argv):
                    platform = sys.argv[i + 1]
                    try:
                        platform = int(platform)
                        deviceSettings.platform = platform
                        deviceSettings.useSettings = True
                        i += 1  # Skip the next argument as it's the platform
                    except ValueError:
                        print(f'--platform must be one of the following, not "{platform}":')
                        for idx, plat in enumerate(cl.get_platforms()):
                            print(f"{idx}: {plat}")
                        exit(1)
                else:
                    print("--platform requires an argument")
                    exit(1)
            case "--device":
                # If a device is specified, get the next argument and set it as the device
                if i + 1 < len(sys.argv):
                    device = sys.argv[i + 1]
                    try:
                        device = int(device)
                        deviceSettings.device = device
                        deviceSettings.useSettings = True
                        i += 1  # Skip the next argument as it's the device
                    except ValueError:
                        print(f'--device must be an integer, not "{device}"')
                        exit(1)
                else:
                    print("--device requires an argument")
                    exit(1)
            case "--help":
                print("Help page:")
                print("  --cpu                 : Run the program on CPU")
                print("  --gpu                 : Run the program on GPU")
                print("  --platform [PLATFORM] : Select which platform the device is on")
                print("  --device [DEVICE]     : Select the device to run on")
                exit(0)
        i += 1  # Move to the next argument

    # If creating a context is required, do it
    if settings.calcMethod == "gpu":
        # If settings are to be used, fill in defaults where needed
        if deviceSettings.useSettings:
            if deviceSettings.platform is None:
                deviceSettings.platform = deviceSettings.defaultPlatform
            if deviceSettings.device is None:
                deviceSettings.device = deviceSettings.defaultDevice

        settings.context = createContext()
    main()

