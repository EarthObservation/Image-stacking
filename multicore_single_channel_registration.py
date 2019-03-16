
import warnings
import numpy as np
import time
from osgeo import gdal_array
import imreg_dft as ird
from osgeo import gdal
import scipy.misc
import scipy as sp
import matplotlib.pyplot as plt
import os
import glob
import multiprocessing
from tqdm import tqdm
threads = 4
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

files_path = "/home/juren/Dropbox/space.si/1. naloga/examples/multicore/multicore_single_channel/frames/"

start = time.time()
# List all .tif images in selected folder
paths = sorted(glob.glob(files_path + "*.tif"))
number_of_files = len(paths)
print("Number of .tif files found: " + str(len(paths)))
print("\n")
print("File list:")
for i in range(len(paths)):
    print(paths[i])
"""
Loading the first image to get the image dimensions for definition of arrays
"""
im = gdal_array.LoadFile(paths[0])

"""
Every CHx array represents a color channel of all images
"""
CH0 = np.zeros((number_of_files, len(
    im[0, :, 0]), len(im[0, 0, :])), dtype=np.int16)
CH1 = np.zeros((number_of_files, len(
    im[0, :, 0]), len(im[0, 0, :])), dtype=np.int16)
CH2 = np.zeros((number_of_files, len(
    im[0, :, 0]), len(im[0, 0, :])), dtype=np.int16)
CH3 = np.zeros((number_of_files, len(
    im[0, :, 0]), len(im[0, 0, :])), dtype=np.int16)

CH0[0] = im[0]
CH1[0] = im[1]
CH2[0] = im[2]
CH3[0] = im[3]

result_image = np.zeros(
    (4, len(im[0, :, 0]), len(im[0, 0, :])), dtype=np.int16)

"""
Writing image date to the 4 channels arrays
"""
for i in range(1, number_of_files):
    im = gdal_array.LoadFile(paths[i])
    CH0[i] = im[0]
    CH1[i] = im[1]
    CH2[i] = im[2]
    CH3[i] = im[3]
stop = time.time()
print(stop - start)


"""
Function for the image registration
"""


def part_channel_reg(start_id, stop_id, template_id,
                     channel_id, CH_array, writer, corenumber):
    """writes the selected template image to the results folder"""
    gdal_array.SaveArray(
        CH_array[template_id],
        "results/" +
        str(channel_id) +
        "/" +
        str(channel_id) +
        "template_gdalsave.tif",
        format="GTiff",
        prototype=None)
    for i in tqdm(range(start_id, stop_id), position=corenumber):
        if i != template_id:
            """ird.similarity return the transformation data for every image
            basend on the template image (translation vector, rotation angle
            and scale"""
            result = ird.similarity(
                CH_array[template_id],
                CH_array[i],
                numiter=1,
                order=1)
            """writer SHOULD write to log file, not working at moment due to
            multiprocessing"""
            writer.write(str(result['tvec']) +
                         " " +
                         str(result['angle']) +
                         " " +
                         str(result['scale']) +
                         " " +
                         str(result['success']) +
                         "\n")
            img_transformed = ird.transform_img(
                CH_array[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=3)
            """transforms the image from the selected channel based on the
            transformation data"""
            gdal_array.SaveArray(
                img_transformed,
                "results/" +
                str(channel_id) +
                "/" +
                str(channel_id) +
                "_" +
                str(i).zfill(2) +
                "gdalsave.tif",
                format="GTiff",
                prototype=None)
            """saves the transformed image"""
            for j in range(4):
                if j != channel_id:
                    if j == 0:
                        img = CH0[i]
                    if j == 1:
                        img = CH1[i]
                    if j == 2:
                        img = CH2[i]
                    if j == 3:
                        img = CH3[i]
                    """transformes images for other 3 channels and saves them"""
                    img_transformed = ird.transform_img(
                        img,
                        scale=result['scale'],
                        angle=result['angle'],
                        tvec=result['tvec'],
                        mode='constant',
                        bgval=0,
                        order=1)
                    gdal_array.SaveArray(
                        img_transformed,
                        "results/" +
                        str(j) +
                        "/" +
                        str(j) +
                        "_" +
                        str(i).zfill(2) +
                        "gdalsave.tif",
                        format="GTiff",
                        prototype=None)


def single_channel_average(result_filepath, channel):
    """ images to be avereged  in the results folder """
    paths = sorted(glob.glob(result_filepath + str(channel) + "/*.tif"))
    number_of_files = len(paths)
    im = gdal_array.LoadFile(paths[0])
    CH = np.zeros((number_of_files, len(
        im[:, 0]), len(im[0, :])), dtype=np.int16)
    CH[0] = im

    for i in range(1, number_of_files):
        im = gdal_array.LoadFile(paths[i])
        CH[i] = im
    CH_avg = np.zeros((len(im[:, 0]), len(im[0, :])), dtype=np.int16)
    for i in tqdm(range(len(im[0, :])), position=channel):
        for j in range(len(im[:, 0])):
            suma = 0

            for k in range(number_of_files):
                if CH[k, i, j] == 0:
                    CH_avg[i, j] = 0
                    break
                else:
                    suma = suma + CH[k, i, j]

            CH_avg[i, j] = suma / number_of_files

    """saves the averege result image for the single channel"""
    gdal_array.SaveArray(
        CH_avg,
        "results/avg/" +
        str(channel).zfill(2) +
        "average_gdalsave.tif",
        format="GTiff",
        prototype=None)


files_per_core = int(number_of_files / 4)
result_filepath = "/home/juren/Dropbox/space.si/1. naloga/examples/multicore/multicore_single_channel/results/"

"""main function --- start of multiprocessing"""
if __name__ == '__main__':
    with open("test.txt", "a") as writer:
        channel_id = 3
        template_id = 3
        jobs = []
        if channel_id == 0:
            CH_array = CH0
        if channel_id == 1:
            CH_array = CH1
        if channel_id == 2:
            CH_array = CH2
        if channel_id == 3:
            CH_array = CH3
        for i in range(4):
            if i == 0:
                start_id = files_per_core * i
                stop_id = files_per_core * (i + 1)
            if i == 1:
                start_id = files_per_core * i
                stop_id = files_per_core * (i + 1)
            if i == 2:
                start_id = files_per_core * i
                stop_id = files_per_core * (i + 1)
            if i == 3:
                start_id = files_per_core * i
                stop_id = number_of_files

            """for i in range 4 (number of channels) starts a process on a new
            core for new color channel"""
            p = multiprocessing.Process(
                target=part_channel_reg,
                args=(
                    start_id,
                    stop_id,
                    template_id,
                    channel_id,
                    CH_array,
                    writer,
                    i))
            jobs.append(p)
        #    print("reg on core:", i)
            p.start()
        middle_time = time.time()
        for p in jobs:
            p.join()
        jobs2 = []
        for i in range(4):
            """starts the processes for averaging every channel"""
            p = multiprocessing.Process(
                target=single_channel_average, args=(
                    result_filepath, i))
            jobs2.append(p)
            p.start()
        for p in jobs2:
            p.join()
    writer.close()
stop1 = time.time()

avg_filepath = result_filepath + "avg/"
avg_images_paths = glob.glob(avg_filepath + "*.tif")
avg_images_paths.sort()

print("Time after averaging: " + str(stop1 - start))

print(avg_images_paths)
"""opens the 4 averaged images and combines them in a single 4 channel image"""
for i in range(len(avg_images_paths)):

    result_image[i] = gdal_array.LoadFile(avg_images_paths[i])
result_image = result_image.astype(np.int16)

gdal_array.SaveArray(
    result_image,
    "results/avg/end/result_image_gdalsave.tif",
    format="GTiff",
    prototype=None)

stop2 = time.time()
print("End time: " + str(stop2 - start))
