#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:37:33 2019

@author: juren
"""


import multiprocessing
from tqdm import tqdm
import glob
import os
import imreg_dft as ird
from osgeo import gdal_array
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

files_path = "frames/"

start = time.time()
# List all .tif images in selected folder
paths = sorted(glob.glob(files_path + "*.tif"))
number_of_files = len(paths)
print("Number of .tif files found: " + str(len(paths)))
print("\n")
print("File list:")
for i in range(len(paths)):
    print(os.path.basename(paths[i]))
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

"""
Shared 1D array, one per channel
"""
CH0mp = multiprocessing.RawArray(
    "d", number_of_files * len(im[0, :, 0]) * len(im[0, 0, :]))
CH1mp = multiprocessing.RawArray(
    "d", number_of_files * len(im[0, :, 0]) * len(im[0, 0, :]))
CH2mp = multiprocessing.RawArray(
    "d", number_of_files * len(im[0, :, 0]) * len(im[0, 0, :]))
CH3mp = multiprocessing.RawArray(
    "d", number_of_files * len(im[0, :, 0]) * len(im[0, 0, :]))

"""
Shared array for the average results
"""
AVGmp = multiprocessing.RawArray("d", 4 * len(im[0, :, 0]) * len(im[0, 0, :]))


for i in range(1, number_of_files):
    im = gdal_array.LoadFile(paths[i])
    CH0[i] = im[0]
    CH1[i] = im[1]
    CH2[i] = im[2]
    CH3[i] = im[3]
stop = time.time()
print(stop - start)

image_1d_size = len(im[0, :, 0]) * len(im[0, 0, :])


print(CH0mp[8500000])
print(CH1mp[8500000])
print(CH2mp[8500000])
print(CH3mp[8500000])


def part_channel_reg(
        start_id,
        stop_id,
        template_id,
        channel_id,
        CH_array,
        core_number,
        CH0mp,
        CH1mp,
        CH2mp,
        CH3mp):

    img_start = template_id * len(im[0, :, 0]) * len(im[0, 0, :])
    img_stop = img_start + len(im[0, :, 0]) * len(im[0, 0, :])
    """
    saves the templage image for evry channel
    """
    if core_number == 0:
        CH0mp[img_start:img_stop] = np.reshape(CH0[template_id], image_1d_size)
    if core_number == 1:
        CH1mp[img_start:img_stop] = np.reshape(CH0[template_id], image_1d_size)
    if core_number == 2:
        CH2mp[img_start:img_stop] = np.reshape(CH0[template_id], image_1d_size)
    if core_number == 3:
        CH3mp[img_start:img_stop] = np.reshape(CH0[template_id], image_1d_size)

    for i in tqdm(range(start_id, stop_id), position=core_number):
        if i != template_id:
            """ird.similarity return the transformation data for every image
            basend on the template image (translation vector, rotation angle
            and scale"""
            img_start = i * image_1d_size
            img_stop = img_start + image_1d_size
            result = ird.similarity(
                CH_array[template_id],
                CH_array[i],
                numiter=1,
                order=1)

            """transforms the image from the selected channel based on the
            transformation data"""
            img_transformed = ird.transform_img(
                CH0[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=3)
            CH0mp[img_start:img_stop] = np.reshape(
                img_transformed, image_1d_size)
            img_transformed = ird.transform_img(
                CH1[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=3)
            CH1mp[img_start:img_stop] = np.reshape(
                img_transformed, image_1d_size)
            img_transformed = ird.transform_img(
                CH2[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=3)
            CH2mp[img_start:img_stop] = np.reshape(
                img_transformed, image_1d_size)
            img_transformed = ird.transform_img(
                CH3[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=3)
            CH3mp[img_start:img_stop] = np.reshape(
                img_transformed, image_1d_size)


def single_channel_average(channel, core_number, AVGmp):
    #    for i in  tqdm(range(image_1d_size), position=core_number):
    # for i in range(image_1d_size):
    #        count=0
    #        suma=0
    #        for j in range(number_of_files):
    #            if channel[j*image_1d_size]>0:
    #                suma=suma+channel[j*image_1d_size]
    #                count=count+1
    #        if count==number_of_files:
    #            channel[i]=suma/number_of_files
    #        else:
    #            channel[i]=0

    #    CH=np.reshape(channel,(number_of_files,len(im[0,:,0]),len(im[0,0,:])))

    CH_avg = np.zeros((len(im[0, :, 0]), len(im[0, 0, :])))
    CH = np.frombuffer(channel).reshape(
        (number_of_files, len(im[0, :, 0]), len(im[0, 0, :])))
    for i in tqdm(range(len(im[0, :, 0])), position=core_number):
        for j in range(len(im[0, 0, :])):
            suma = 0
            count = 0
            for k in range(number_of_files):
                if CH[k, i, j] > 0:
                    count = count + 1

                suma = suma + CH[k, i, j]

            if count == number_of_files:
                CH_avg[i, j] = suma / number_of_files
            else:
                CH_avg[i, j] = 0

    gdal_array.SaveArray(
        CH_avg,
        str(core_number) +
        "test_test_gdalsave.tif",
        format="GTiff",
        prototype=None)

    img_start = core_number * image_1d_size
    img_stop = img_start + image_1d_size
    AVGmp[img_start:img_stop] = np.reshape(CH_avg, image_1d_size)


files_per_core = int(number_of_files / 4)

if __name__ == '__main__':

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
                i,
                CH0mp,
                CH1mp,
                CH2mp,
                CH3mp,
            ))
        jobs.append(p)
    #    print("reg on core:", i)
        p.start()
    middle_time = time.time()
    for p in jobs:
        p.join()

    jobs_avg = []

    for k in range(4):
        if k == 0:
            channel = CH0mp
        if k == 1:
            channel = CH1mp
        if k == 2:
            channel = CH2mp
        if k == 3:
            channel = CH3mp
        q = multiprocessing.Process(
            target=single_channel_average, args=(
                channel, k, AVGmp))
        jobs_avg.append(q)
        q.start()
    for q in jobs_avg:
        q.join()

#   The images are transformed back from the 1d shared array to 2d
    CH0_avg = np.reshape(AVGmp[0:image_1d_size],
                         (len(im[0, :, 0]), len(im[0, 0, :])))
    CH1_avg = np.reshape(
        AVGmp[image_1d_size:2 * image_1d_size], (len(im[0, :, 0]), len(im[0, 0, :])))
    CH2_avg = np.reshape(
        AVGmp[2 * image_1d_size:3 * image_1d_size], (len(im[0, :, 0]), len(im[0, 0, :])))
    CH3_avg = np.reshape(
        AVGmp[3 * image_1d_size:4 * image_1d_size], (len(im[0, :, 0]), len(im[0, 0, :])))

#   Cropping the black part of the images
    non_zero_image = CH0_avg > 0
    coords = np.argwhere(non_zero_image)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

# Left and right border manual crop
    x0 = x0 + 100
    x1 = x1 - 100
# top and bottom border manual crop
    y0 = y0 + 50
    y1 = y1 - 50
    CH1234 = np.empty((4, y1 - y0, x1 - x0), dtype=np.int16)
    print((len(CH1234[0, :, 0]), len(CH1234[0, 0, :])))

    CH1234[0] = CH0_avg[y0:y1, x0:x1]
    CH1234[1] = CH1_avg[y0:y1, x0:x1]
    CH1234[2] = CH2_avg[y0:y1, x0:x1]
    CH1234[3] = CH3_avg[y0:y1, x0:x1]

#   Saving the images four one per channel and one combined
    gdal_array.SaveArray(
        CH1234[0],
        "CH0_average.tif",
        format="GTiff",
        prototype=None)
    gdal_array.SaveArray(
        CH1234[1],
        "CH1_average.tif",
        format="GTiff",
        prototype=None)
    gdal_array.SaveArray(
        CH1234[2],
        "CH2_average.tif",
        format="GTiff",
        prototype=None)
    gdal_array.SaveArray(
        CH1234[3],
        "CH3_average.tif",
        format="GTiff",
        prototype=None)
    gdal_array.SaveArray(
        CH1234,
        "CH1234_average.tif",
        format="GTiff",
        prototype=None)
