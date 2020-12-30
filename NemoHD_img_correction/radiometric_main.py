#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of usage of the NemoHD_tools.py
@author: Jure Nemec
Input: work_folder

Process:
    
    
    - creating of subdirectories:
        - /corr/
        - /corr/stacked
        - /corr/rgb
    - transformation from raw to tif
    - single channel tifs listing and sorting for additional use
    - RGB flaftield correction
    - Nir flatfield correction
    - RGBN stacking:
        Each channel is stacked separately
    - RGB channels are merged in a single file
    - RGB channels in the single file are aligned
    - jpg prewievs are created


"""


import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from osgeo import gdal_array
import gdal
import glob
import time
# import imreg_dft as ird
import os
from affine import Affine
from scipy.io.idl import readsav
import rasterio
from rasterio.merge import merge
from itertools import permutations 
from tqdm import tqdm
import inquirer
from scipy import ndimage
import multiprocessing as mp
import configparser
import NemoHD_tools as NHD
from scipy import stats
import natsort


t0=time.time()

start=time.time()


"""
definition of the workfolder with the raw single channel files
"""
work_folder="/home/juren/space_ext/pakistan/cele/test2/"



"""
adding .raw extension to raw files
"""
files=glob.glob(work_folder+"*")

if files[0][-4:]!=".raw":

    for file in files:
       os.rename(file, file+".raw")

f = open(work_folder+"time_log.txt", "a")

"""
creation of sub directories in the workfolder
"""
if not os.path.exists(work_folder+"corr/"):
    os.makedirs(work_folder+"corr/")

if not os.path.exists(work_folder+"corr/stacked/"):
    os.makedirs(work_folder+"corr/stacked/")
    
if not os.path.exists(work_folder+"corr/rgb/"):
    os.makedirs(work_folder+"corr/rgb/")



"""
conversion of raw files to .tif
"""
raws=glob.glob(work_folder+"*.raw")

for file in raws:
    CH=os.path.basename(file)[3]    
    NHD.raw_to_tif(file, CH)

f.write("Conversion to raw: "+str(time.time()-start))

"""
listing of single channel tiffs
"""
R=NHD.single_channel_tifs(work_folder, "R")
G=NHD.single_channel_tifs(work_folder, "G")
B=NHD.single_channel_tifs(work_folder, "B")

print(R, G, B)
R.sort()
G.sort()
B.sort()

N=NHD.single_channel_tifs(work_folder, "N")

start=time.time()


"""
RGB flatfield correction
"""
for i in range(len(R)):
    NHD.flatfield_corr(R[i], G[i], B[i])
    
    
    
"""
Nir flatfield correction
"""
for tif in N:
    NHD.NIR_flatfield_corr(tif)
    
f.write("Flatfield correction: "+str(time.time()-start))    
start=time.time()

"""
RGB_stacking
"""
channels=["B", "G", "R", "N"]

# N.sort()
# print(N)
# channels=["N"]
for channel in channels:
    tifs=glob.glob(work_folder+"corr/*.tif")
    channel_tifs=[]
    for tif in tifs:
        # print(os.path.basename(tif))
        if os.path.basename(tif)[3]==channel:
            channel_tifs.append(tif)
        
    
    # channel_tifs.sort(key = lambda x: x.split("_")[-1][:-5])
    """
    natsort: natural/human sorting  ...1D comes before ...10D for the correct stacking order
    """
    channel_tifs=natsort.natsorted(channel_tifs)
    print(channel_tifs)
    NHD.single_channel_stacking_unlimited(channel_tifs)



f.write("stacking: "+str(time.time()-start))    

"""
joining/stacking stacked channels
"""
tifs=glob.glob(work_folder+"corr/stacked/*.tif")
for tif in tifs:
    if os.path.basename(tif)[3]=="R":
        R=tif
    if os.path.basename(tif)[3]=="G":
        G=tif
    if os.path.basename(tif)[3]=="B":
        B=tif
    if os.path.basename(tif)[3]=="N":
        N=tif


rgb_file=NHD.single_to_rgb(R, G, B)

"""
Single image RGB channel aligning
"""
NHD.single_image_band_match(rgb_file)



"""
creating prewiev jpgs
"""

tifs=glob.glob(work_folder+"corr/stacked/*.tif")
for tif in tifs:
    NHD.create_prewiev_jpg(tif)

f.write("complete_time: "+str(time.time()-t0))    

f.close()