#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:37:33 2019

@author: juren
"""


import sys
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

if len(sys.argv) < 4:
    print("Parameters missing not set. Add paramaters.")
    print(
        sys.argv[0] +
        " template_id channel_id averaging_mode file_name(optional).")
    print("\n")
    print(
        "template_id: Index (integer) of the template image for the registration. From 0 to " +
        str(number_of_files) +
        ".")
    print("channel_id: Index (integer) of the choosen channel for image registration From 0 to 3.")
    print("Averaging mode: ")
    print("cmp  Returns the full image with partial to full overlaying.")
    print("crp  Return the cropped image, only where all " +
          str(number_of_files) + " images overlay.")
    print("file_name: Text mode. Base filename of the textfiles of the transformation results (translation, scale, rotation). If no filename is given, writing is disabled.")

else:
    """
    Loading the first image to get the image dimensions for definition of arrays
    """
    template_id = int(sys.argv[1])
    channel_id = int(sys.argv[2])

    im = gdal_array.LoadFile(paths[template_id])

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
    Y0 = multiprocessing.Value("i", 0)
    Y1 = multiprocessing.Value("i", 9999)

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
    AVGmp = multiprocessing.RawArray(
        "d", 4 * len(im[0, :, 0]) * len(im[0, 0, :]))

    i = 1
    while i < number_of_files:
        if i <= template_id:
            j = i - 1
        else:
            j = i
        print(i)
        print(paths[j])
        im = gdal_array.LoadFile(paths[j])
        CH0[i] = im[0]
        CH1[i] = im[1]
        CH2[i] = im[2]
        CH3[i] = im[3]
        i = i + 1

    stop = time.time()
    print("File reading time:")
    print(stop - start)
    print("\n")

    image_1d_size = len(im[0, :, 0]) * len(im[0, 0, :])

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
        if len(sys.argv) == 5:
            writer = open(
                str(template_id) +
                "_" +
                str(channel_id) +
                "_" +
                sys.argv[3] +
                "_" +
                sys.argv[4] +
                "_" +
                str(start_id) +
                "_" +
                str(stop_id) +
                "_.txt",
                "a")
            writer.write("template_id: " + str(template_id) + "\n")
            writer.write("channel_id: " + str(channel_id) + "\n")
            writer.write("scale rotation translation success \n")

        img_start = 0
        img_stop = img_start + len(im[0, :, 0]) * len(im[0, 0, :])
        """
        saves the templage image for evry channel
        """
        if core_number == 0:
            CH0mp[img_start:img_stop] = np.reshape(CH0[0], image_1d_size)
        if core_number == 1:
            CH1mp[img_start:img_stop] = np.reshape(CH0[0], image_1d_size)
        if core_number == 2:
            CH2mp[img_start:img_stop] = np.reshape(CH0[0], image_1d_size)
        if core_number == 3:
            CH3mp[img_start:img_stop] = np.reshape(CH0[0], image_1d_size)

        for i in tqdm(range(start_id, stop_id), position=core_number):

            """ird.similarity return the transformation data for every image
            basend on the template image (translation vector, rotation angle
            and scale"""
            img_start = i * image_1d_size
            img_stop = img_start + image_1d_size
            result = ird.similarity(
                CH_array[0], CH_array[i], numiter=3, order=3)
            if len(sys.argv) == 5:
                writer.write(str(i) +
                             " " +
                             str(result['scale']) +
                             " " +
                             str(result['angle']) +
                             " " +
                             str(result['tvec']) +
                             " " +
                             str(result['success']) +
                             "\n")

            """transforms the image from the selected channel based on the
            transformation data"""
            img_transformed = ird.transform_img(
                CH0[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=1)
            CH0mp[img_start:img_stop] = np.reshape(
                img_transformed, image_1d_size)
            img_transformed = ird.transform_img(
                CH1[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=1)
            CH1mp[img_start:img_stop] = np.reshape(
                img_transformed, image_1d_size)
            img_transformed = ird.transform_img(
                CH2[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=1)
            CH2mp[img_start:img_stop] = np.reshape(
                img_transformed, image_1d_size)
            img_transformed = ird.transform_img(
                CH3[i],
                scale=result['scale'],
                angle=result['angle'],
                tvec=result['tvec'],
                mode='constant',
                bgval=0,
                order=1)
            CH3mp[img_start:img_stop] = np.reshape(
                img_transformed, image_1d_size)
        if len(sys.argv) == 5:
            writer.close()

    def single_channel_average(channel, core_number, AVGmp, Y0, Y1):
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
        if sys.argv[3] == 'cmp':
            for i in tqdm(range(len(im[0, :, 0])), position=core_number):
                for j in range(len(im[0, 0, :])):
                    suma = 0
                    count = 0
                    for k in range(number_of_files):
                        if CH[k, i, j] > 0:
                            count = count + 1

                        suma = suma + CH[k, i, j]

                    if count > 0:
                        CH_avg[i, j] = suma / count
                    else:
                        CH_avg[i, j] = 0

        if sys.argv[3] == 'crp':
            y0 = 0
            y1 = len(im[0, 0, :])
            for i in tqdm(range(number_of_files), position=core_number):
                CH_avg = CH_avg + CH[i]
                non_zero_image = CH[i] > 0
                coords = np.argwhere(non_zero_image)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1
                if y0 > Y0.value:
                    Y0.value = y0
                if y1 < Y1.value:
                    Y1.value = y1
            CH_avg = CH_avg / number_of_files

        img_start = core_number * image_1d_size
        img_stop = img_start + image_1d_size
        AVGmp[img_start:img_stop] = np.reshape(CH_avg, image_1d_size)

    files_per_core = int(number_of_files / 4)

    if __name__ == '__main__':
        print("template_id: " + str(template_id))
        print("channel_id: " + str(channel_id))
        print("Selected averaging mode: " + sys.argv[3] + "\n")
        if len(sys.argv) > 4:
            print("Text mode is ON")
        else:
            print("Text mode is OFF")
        reg_start = time.time()
        print("Image registration on 4 cores... \n")
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
                start_id = 1 + files_per_core * i
                stop_id = 1 + files_per_core * (i + 1)
            if i == 1:
                start_id = 1 + files_per_core * i
                stop_id = 1 + files_per_core * (i + 1)
            if i == 2:
                start_id = 1 + files_per_core * i
                stop_id = 1 + files_per_core * (i + 1)
            if i == 3:
                start_id = 1 + files_per_core * i
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
        reg_stop = time.time()
        print("\n \n \n \n")
        print("Registration time:")
        print(reg_stop - reg_start)
        print("Image averaging... \n")
        jobs_avg = []
        avg_start = time.time()
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
                    channel, k, AVGmp, Y0, Y1))
            jobs_avg.append(q)
            q.start()
        for q in jobs_avg:
            q.join()
        avg_stop = time.time()
        print("\n \n \n \n")
        print("Averaging time: ")
        print(avg_stop - avg_start)
    #   The images are transformed back from the 1d shared array to 2d
        CH0_avg = np.reshape(AVGmp[0:image_1d_size],
                             (len(im[0, :, 0]), len(im[0, 0, :])))
        CH1_avg = np.reshape(
            AVGmp[image_1d_size:2 * image_1d_size], (len(im[0, :, 0]), len(im[0, 0, :])))
        CH2_avg = np.reshape(
            AVGmp[2 * image_1d_size:3 * image_1d_size], (len(im[0, :, 0]), len(im[0, 0, :])))
        CH3_avg = np.reshape(
            AVGmp[3 * image_1d_size:4 * image_1d_size], (len(im[0, :, 0]), len(im[0, 0, :])))

        if sys.argv[3] == 'crp':

            # Left and right border manual crop
            x0 = 100
            x1 = len(im[0, :, 0]) - 100
    # top and bottom border manual crop
            y0 = Y0.value + 50
            y1 = Y1.value - 50
            CH1234 = np.empty((4, y1 - y0, x1 - x0), dtype=np.int16)
            CH1234[0] = CH0_avg[y0:y1, x0:x1]
            CH1234[1] = CH1_avg[y0:y1, x0:x1]
            CH1234[2] = CH2_avg[y0:y1, x0:x1]
            CH1234[3] = CH3_avg[y0:y1, x0:x1]

            print("Saving images...")
            gdal_array.SaveArray(
                CH1234[0],
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH0_crp_average.tif",
                format="GTiff",
                prototype=None)
            gdal_array.SaveArray(
                CH1234[1],
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH1_crp_average.tif",
                format="GTiff",
                prototype=None)
            gdal_array.SaveArray(
                CH1234[2],
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH2_crp_average.tif",
                format="GTiff",
                prototype=None)
            gdal_array.SaveArray(
                CH1234[3],
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH3_crp_average.tif",
                format="GTiff",
                prototype=None)
            gdal_array.SaveArray(
                CH1234,
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH1234_crp_average.tif",
                format="GTiff",
                prototype=None)
        else:
            CH1234 = np.empty(
                (4, len(im[0, :, 0]), len(im[0, 0, :])), dtype=np.int16)
            CH1234[0] = CH0_avg
            CH1234[1] = CH1_avg
            CH1234[2] = CH2_avg
            CH1234[3] = CH3_avg

    #   Saving the images four one per channel and one combined
            print("Saving images...")
            gdal_array.SaveArray(
                CH1234[0],
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH0_cmp_average.tif",
                format="GTiff",
                prototype=None)
            gdal_array.SaveArray(
                CH1234[1],
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH1_cmp_average.tif",
                format="GTiff",
                prototype=None)
            gdal_array.SaveArray(
                CH1234[2],
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH2_cmp_average.tif",
                format="GTiff",
                prototype=None)
            gdal_array.SaveArray(
                CH1234[3],
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH3_cmp_average.tif",
                format="GTiff",
                prototype=None)
            gdal_array.SaveArray(
                CH1234,
                str(template_id) +
                "_" +
                str(channel_id) +
                "_CH1234_cmp_average.tif",
                format="GTiff",
                prototype=None)
        stop2 = time.time()
        print("Completed process after:")
        print(stop2 - start)
