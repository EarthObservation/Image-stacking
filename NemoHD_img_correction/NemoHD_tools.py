#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NemoHD_tools.py

Different functions and tools for image processing of the NemoHD imagery

@author: Jure Nemec
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from osgeo import gdal_array
import gdal
import glob
import imreg_dft as ird
import os
from affine import Affine
from scipy import ndimage
import osgeo
from astropy.io import fits



def resample_raster(raster, src_xres, dst_xres, src_yres,dst_yres):
    
    new_raster=ndimage.zoom(raster,[dst_yres/src_yres, dst_xres/src_xres], order=3)
    return new_raster

def array_poduct(a, b):
    # faktor = np.sqrt(np.sum(np.square(a - b)))
    faktor=[a_* b_ for a_, b_ in zip(a, b)]
    return faktor

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



def simple_flatfield_corr(flat, real, Mfaktor, Pfaktor):
    """
simple_flatfield_corr

flatfield correction based on the average of a flat array

Parameters: flat:array
            real:array
            Mfaktor:float
            Pfaktor:float


Returns:    C:array

    The flatfield corrected array
"""
    real=real.astype(np.float64)
    flat=flat.astype(np.float64)
    # init_max=np.max(real)/4095
    print(np.max(real))
    flat=NormalizeData(flat)
    flat=flat*Mfaktor   
    m=np.average(flat)
    gain=m/(flat+Pfaktor)   
    C=np.multiply(gain, real)    
    print(np.max(C))
    return C


def BLUE_simple_flatfield_corr(flat, real):
    """
BLUE_simple_flatfield_corr

flatfield correction based on the average of the healthy part of the flat array

Parameters: flat:array
            real:array


Returns:    C:array

    The flatfield corrected array
"""
    real=real.astype(np.float64)
    flat=flat.astype(np.float64)

    
    row_averages=[]
    for row in real:
        row_median=np.median(row)
        row_averages.append(row_median)
    plt.plot(np.linspace(0,len(row_averages), len(row_averages)),row_averages)
    plt.show()
    plt.close()
    
    row_averages=[]
    for row in flat:
        row_median=np.median(row)
        row_averages.append(row_median)
    plt.plot(np.linspace(0,len(row_averages), len(row_averages)),row_averages, color="red")
    
    
    target_average=np.median(row_averages[700:])
    F_row_average=row_averages.copy()
    k=(row_averages[750]-row_averages[1500])/750
    
    for i in range(1501):
        F_row_average[1500-i]=row_averages[1500]+k*i
    for i in range(len(row_averages)-1500):
        F_row_average[1500+i]=row_averages[1500]-k*i
    plt.plot(np.linspace(0,len(row_averages), len(row_averages)),F_row_average, color="blue")
    
    C=np.copy(real)
    for i, row in enumerate(flat, start=0):
        row_median=np.median(row)
        row_averages[i]=row_averages[i]/(row_median/target_average)
        gain=row_median/F_row_average[i]
        C[i]=C[i]/gain
    plt.plot(np.linspace(0,len(row_averages), len(row_averages)),row_averages, color="green")  
    plt.show()
    plt.close()
    
    return C


def raw_to_tif(file, channel=None ):
    """
raw_to_tif

transformation of raw files to tif files

Parameters: files:str
                filepath to raw file (strings)
            
            CH:str
                channel (R,G,B,N,P)

Returns:    None

    The tif files are saved in the same locatiion and name with added .tif 
    extension
"""
    
    def read_uint12(data_chunk):
        data = np.frombuffer(data_chunk, dtype=np.uint8)
        fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
        # fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
        # snd_uint12 = (lst_uint8 << 4) + (np.bitwise_and(15, mid_uint8))
        fst_uint12 = (fst_uint8 << 4) + (np.bitwise_and(15, mid_uint8))
        snd_uint12 = (lst_uint8 << 4) + (mid_uint8 >> 4)
        return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])

# def read_uint12(data_chunk):
#     data = np.frombuffer(data_chunk, dtype=np.uint8)
#     fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
#     fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
#     snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
#     return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])

# def read_uint12(data_chunk):
#     data = np.frombuffer(data_chunk, dtype=np.uint8)
#     fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
#     fst_uint12 = ((mid_uint8 & 0x0F) << 8) | fst_uint8
#     snd_uint12 = (lst_uint8 << 4) | ((mid_uint8 & 0xF0) >> 4)
#     return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    
    # infile = 'd:\\Projekti\\Satelit\\CO\\Razpis\\Flat field images_new2020\\flatfield\\NHDBflat_1D'
    # infile = 'd:\Projekti\Satelit\CO\Razpis\_POSNETKI\Jure_naloga_banje_raw_pyt\\NHDRGoreMorje_3D'

    # in_path = 'p:\\NEMO\Posnetki\\20201014_GoreMorje_data\cele\\'
    # in_path = 'd:\Projekti\Satelit\CO\Razpis\_POSNETKI\Peking_PAN\\'
    # in_image_files = [filename for filename in os.listdir(in_path) if filename.lower().startswith("nhd") and filename.lower().endswith("d")]

    
    # infile = in_path + in_image_files[i]
    with open(file, 'rb', buffering=10) as f:  # problem pri branju podatkov?
        byte = f.read()
    print(file)
    # # ar = open(infile, 'rb')
    # buffer = BytesIO()
    # byte = BytesIO(ar)
    
    img = read_uint12(byte)
    print(img)
    
    if channel=="P":
        img = img.reshape((2748, 3664))  # PAN
    else:
        img = img.reshape((2050, 2448))  # MS
    # img = img.reshape((2748, 3664))  # PAN

    size = img.shape
    
    
    out = file[:-4]+ "_py.tif"

    driver = gdal.GetDriverByName('GTiff')

    outRaster = driver.Create(out, size[1], size[0], 1, gdal.GDT_UInt16)

    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(img)
    outband.FlushCache()


def single_to_rgb(R_file,G_file,B_file):
    """
single_to_rgb

Merge separate RGB files to an RGB image

Parameters: R_file:str
                filepath of the R channel .tif
            G_file:str
                filepath of the G channel .tif
            B_file:str
                filepath of the B channel .tif
            


Returns:    file_path:str

    The tif file is saved in the same locatiion and name with added _rgb_.tif"
    extension. Also return filepath.
""" 
    R=gdal_array.LoadFile(R_file)
    G=gdal_array.LoadFile(G_file)
    B=gdal_array.LoadFile(B_file)
    
  
    basename=os.path.basename(R_file)
    basename=basename[:3]+basename[4:]
    basename=basename[:-4]+"_rgb_.tif"      
    

    file_path=os.path.dirname(os.path.abspath(R_file))+"/"+basename

    
    driver=osgeo.gdal.GetDriverByName("GTiff")
    options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
    print(file_path)
    print(np.max(np.array([R.shape[1],B.shape[1],G.shape[1]])), np.max(np.array([R.shape[0],B.shape[0],G.shape[0]])))
    Xlen=np.max(np.array([R.shape[1],B.shape[1],G.shape[1]]))
    Ylen= np.max(np.array([R.shape[0],B.shape[0],G.shape[0]]))
    dataset=driver.Create(file_path, int(Xlen),int(Ylen), 3, osgeo.gdal.GDT_UInt16, options)        
    
    dataset.GetRasterBand(1).WriteArray(R)
    dataset.GetRasterBand(2).WriteArray(G)
    dataset.GetRasterBand(3).WriteArray(B)
    
    return file_path

def sort_tifs(folder_path):
    tif_files=glob.glob(folder_path+"*.tif")
    
    for i, file in enumerate(tif_files, start=0):
        file=os.path.basename(file)[:3]+os.path.basename(file)[4:]
        tif_files[i]=file
    tif_files.sort()

    basenames=list(set(tif_files))
    
    
    return basenames


def basename_to_rgb_files(basename, folder_path):
    R=folder_path+basename[:3]+"R"+basename[3:]
    G=folder_path+basename[:3]+"G"+basename[3:]
    B=folder_path+basename[:3]+"B"+basename[3:]
    
    
    
    return R, G, B

def cut_array_border(array):
    """
cut_array_border

Sets one px wide border around the array to 0

Parameters: array:array
                input array
            
Returns:    array:array

    Return the array with  the cut border
    """     
    array[:, [0, array.shape[1]-1]]=0
    array[[0, array.shape[0]-1], :]=0
    
    
    return array

def cut_transformed_array_borders(array):
    """
cut_transformed_array_borders

    Sets one px wide border around the array to 0, the array presents an image 
raster placed in the bigger array    

Parameters: array:array
                input array
            
Returns:    array:array

    Return the array with  the cut border
    """  
    for col in range(array.shape[1]): 
        col_=array[:, col]
        
        where=np.where(col_>0)
        
        if len(where[0])>0:
            
            col_[[np.min(where[0]),np.min(where[0])+1, np.max(where[0]), np.max(where[0])-1 ]]=0
            
            array[:,col]=col_
    
    for row in range(array.shape[0]): 
        row_=array[row,:]
        
        where=np.where(row_>0)
        if len(where[0])>0:

            row_[[np.min(where[0]),np.min(where[0])+1, np.max(where[0]), np.max(where[0])-1 ]]=0
            
            array[row,:]=row_
        
    return array
        
def return_flatfield_set_path(index):
    """
return_flatfield_set_path

    Return    

Parameters: index:int
                range(0-2) for 3 different flatfield sets
                
            
Returns:    flat_file_R:str
            flat_file_G:str
            flat_file_B:str
            flat_file_N:str
            flat_file_P:str

    Returns 5 filepaths to 5 flatfiles, based on the selection in the input
    """ 
    flat_files_R=["/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/new flat field images/NHDRflat_4D.tif",
              "/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/final instrument flat field images/RED-001519920609330501.tif",
              "/home/juren/Dropbox/space.si/NemoHD/image_corr/20201028 Vignetting/flatfield/NHDRflat_4D_py.tif"]
    flat_files_G=["/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/new flat field images/NHDGflat_6D.tif",
                  "/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/final instrument flat field images/GREEN-001519920139565967.tif",
                  "/home/juren/Dropbox/space.si/NemoHD/image_corr/20201028 Vignetting/flatfield/NHDGflat_6D_py.tif"]
    flat_files_B=["/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/new flat field images/NHDBflat_3D.tif",
                  "/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/final instrument flat field images/BLUE-001519920137579544.tif",
                  "/home/juren/Dropbox/space.si/NemoHD/image_corr/20201028 Vignetting/flatfield/NHDBflat_3D_py.tif"]
    flat_files_N=["/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/new flat field images/NHDNflat_5D.tif",
                  "/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/final instrument flat field images/NIR-001519920141573143.tif",
                  "/home/juren/Dropbox/space.si/NemoHD/image_corr/20201028 Vignetting/flatfield/NHDNflat_5D_py.tif"]
    flat_files_P=["/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/new flat field images/NHDPflat_3D.tif",
              "/home/juren/Dropbox/space.si/NemoHD/image_corr/Jure_naloga_popravek_posnetkov/final instrument flat field images/PAN_Flat_Field_20180301_100us.bmp",
              "/home/juren/Dropbox/space.si/NemoHD/image_corr/20201028 Vignetting/flatfield/NHDPflat_3D_py.tif"]



    flat_file_R=flat_files_R[index]
    flat_file_G=flat_files_G[index]
    flat_file_B=flat_files_B[index]
    flat_file_N=flat_files_N[index]
    flat_file_P=flat_files_P[index]
    
    
    return flat_file_R, flat_file_G, flat_file_B, flat_file_N, flat_file_P
        
def NIR_flatfield_corr(tif):
    """
NIR_flatfield_corr

    NIR images flatfield correction    

Parameters: tif:str
                filepath to the tif file
                
            
Returns:    None

    Saves the corrected image at the tif location with added _corr.tif extension
    """ 
    _,_,_,flat_file_N,_=return_flatfield_set_path(2)
    
    print(tif)
    
    N=gdal_array.LoadFile(tif)
    
    file_path=os.path.dirname(os.path.abspath(tif))+"/corr/"+os.path.basename(tif)[:-4]+"_corr.tif"
    
    N=BLUE_simple_flatfield_corr(gdal_array.LoadFile(flat_file_N), N)
    
    options = ['PROFILE=GeoTIFF']
    driver=osgeo.gdal.GetDriverByName("GTiff")
    basename=os.path.basename(tif)
    basename=basename[:3]+"N"+basename[4:]
    
    file_path=os.path.dirname(os.path.abspath(tif))+"/corr/"+basename[:-4]+"_corr.tif"
    dataset=driver.Create(file_path, N.shape[1],N.shape[0],  1, osgeo.gdal.GDT_UInt16, options)        
    dataset.GetRasterBand(1).WriteArray(N)
    dataset.FlushCache()
    dataset = None
    
    

def flatfield_corr(*args):
    """
flatfield_corr

    RGB images flatfield correction    

Parameters: 
        args:
                filepath:str
                    filepath to the RGB tif file
                filepaths:list(strings)
                    filepaths to 3 RGB single tif files in the RGB order
                
            
Returns:    None

    Saves the corrected R G B images at the tif location with added _corr.tif 
    extension and an additional combined RGB image
    """ 
    print(len(args), args)
    if len(args)==1:
        
        raster=gdal_array.LoadFile(args[0])
        R=raster[0]
        G=raster[1]
        B=raster[2]
        file_path=os.path.dirname(os.path.abspath(args[0]))+"/corr/"+os.path.basename(args[0])[:-4]+"_corr.tif"
    elif len(args)==3:
        R=gdal_array.LoadFile(args[0])
        G=gdal_array.LoadFile(args[1])
        B=gdal_array.LoadFile(args[2])
        basename=os.path.basename(args[0])
        basename=basename[:3]+basename[4:]
        file_path=os.path.dirname(os.path.abspath(args[0]))+"/corr/"+os.path.basename(args[0])[:-4]+"_rgb_corr.tif"
    else:
        print("functions takes 1 (rgb file path) or 3 arguments (one filepath per channel)")
    
    
    flat_file_R, flat_file_G, flat_file_B,_,_=return_flatfield_set_path(2)
    
    # Mfaktor=5
    R=BLUE_simple_flatfield_corr(gdal_array.LoadFile(flat_file_R), R)
    G=BLUE_simple_flatfield_corr(gdal_array.LoadFile(flat_file_G), G)
    B=BLUE_simple_flatfield_corr(gdal_array.LoadFile(flat_file_B), B)
    
    G=np.roll(G, 8, axis=1)
    
    # print(np.max(raster), np.min(raster))
    
    
    plot_raster=np.empty([2050, 2448, 3], dtype=np.int16)
    plot_raster[:,:,0]=R 
    plot_raster[:,:,1]=G
    plot_raster[:,:,2]=B
    
    plot_raster=(plot_raster/np.max(plot_raster))*255
    plot_raster=plot_raster.astype(np.uint8)
    plot_raster[:,:,0]=cv2.equalizeHist( plot_raster[:,:,0])
    plot_raster[:,:,1]=cv2.equalizeHist( plot_raster[:,:,1])
    plot_raster[:,:,2]=cv2.equalizeHist( plot_raster[:,:,2])
    
    
    # plt.axis('off')

    
    
    # plt.imshow(plot_raster, vmin=0, vmax=65535)
    # plt.show()
    # plt.close()
    
    # max_=np.max(raster)

    
    # raster=raster.astype(np.uint16)
    
    driver=osgeo.gdal.GetDriverByName("GTiff")
    options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
    
    dataset=driver.Create(os.path.dirname(os.path.abspath(args[0]))+"/corr/rgb/"+os.path.basename(args[0])[:-4]+"_rgb_corr.tif"
                          , R.shape[1]-8,R.shape[0],  3, osgeo.gdal.GDT_UInt16, options)        
    dataset.GetRasterBand(1).WriteArray(R[:,8:])
    dataset.GetRasterBand(2).WriteArray(G[:,8:])
    dataset.GetRasterBand(3).WriteArray(B[:,8:])
    dataset.FlushCache()
    dataset = None
    
    options = ['PROFILE=GeoTIFF']

    basename=os.path.basename(args[0])
    basename=basename[:3]+"R"+basename[4:]
    file_path=os.path.dirname(os.path.abspath(args[0]))+"/corr/"+basename[:-4]+"_corr.tif"
    dataset=driver.Create(file_path, R.shape[1]-8,R.shape[0],  1, osgeo.gdal.GDT_UInt16, options)        
    dataset.GetRasterBand(1).WriteArray(R[:,8:])
    dataset.FlushCache()
    dataset = None
    
    basename=os.path.basename(args[0])
    basename=basename[:3]+"G"+basename[4:]
    file_path=os.path.dirname(os.path.abspath(args[0]))+"/corr/"+basename[:-4]+"_corr.tif"
    dataset=driver.Create(file_path, R.shape[1]-8,R.shape[0],  1, osgeo.gdal.GDT_UInt16, options)        
    dataset.GetRasterBand(1).WriteArray(G[:,8:])
    dataset.FlushCache()
    dataset = None
    
    basename=os.path.basename(args[0])
    basename=basename[:3]+"B"+basename[4:]
    file_path=os.path.dirname(os.path.abspath(args[0]))+"/corr/"+basename[:-4]+"_corr.tif"
    dataset=driver.Create(file_path, R.shape[1]-8,R.shape[0],  1, osgeo.gdal.GDT_UInt16, options)        
    dataset.GetRasterBand(1).WriteArray(B[:,8:])
    dataset.FlushCache()
    dataset = None
    
    
    
    # plot_raster = cv2.cvtColor(plot_raster, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(file_path[:-4]+"_preview.jpg", plot_raster)
    # plot_raster=NormalizeData(plot_raster)
    # plot_raster=plot_raster/4095
    # plot_raster=plot_raster*512
    # plot_raster=plot_raster.astype(np.uint8)
    
    
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    # hspace = 0, wspace = 0)
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    # plt.imshow(plot_raster[:,8:,:])
    # # plt.savefig(file_path[:-4]+"_previewplot.jpg", dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close
    
    
    # # print(img_array.shape)
    
    # # img=Image.fromarray(plot_raster, "RGB")
    # # img.save(file_path[:-4]+"_preview.jpg")
    
    
    
    # # print(np.max(raster), np.min(raster))
    
    
def fits_to_nparray(file):
    """
fits_to_nparray

    fits to tif transformation

Parameters: file:str
                filepath to the .fits file
            
            
Returns:    image_data

    Return the fits image as an array and saves the tif file at the same location 
    
    """
    hdu_list = fits.open(file)
    image_data = hdu_list[0].data
    image_data=image_data.astype(np.uint16)
    
    gdal_array.SaveArray(image_data, file[:-5]+".tif")
    
    return image_data


def pan_corr(file):
    """
pan_corr

    Correction of the pixel values of the PAN images
    by Ales Marsetic

Parameters: file:str
                filepath to the tif file
            
            
Returns:    None

    Saves the correctef tif file at the same location as input
    
    """

    # # infile = 'd:\\Projekti\\Satelit\\CO\\Razpis\\Flat field images_new2020\\flatfield\\NHDBflat_1D'
    # # infile = 'd:\Projekti\Satelit\CO\Razpis\_POSNETKI\Jure_naloga_banje_raw_pyt\\NHDRGoreMorje_3D'
    #
    # # in_path = 'd:\Projekti\Satelit\CO\Razpis\Flat field images_new2020\\20201028 Vignetting\\flatfield\\'
    # # in_pan_ref_file = 'NHDPflat_3D_py.tif'
    # in_path = 'd:\Projekti\Satelit\CO\Razpis\_POSNETKI\Peking_PAN\\'
    # in_pan_ref_file = 'NHDPfoc_swp6_1D_py.tif'
    # in_ref = in_path + in_pan_ref_file
    #
    # inreffil = gdal.Open(in_ref)
    # image_ref = inreffil.ReadAsArray()
    # # size_ref = image_ref.shape
    # # pix_count = size_ref[0]*size_ref[1]
    #
    # image_ref = image_ref[800:930, 1420:1640]
    # size_ref = image_ref.shape
    # pix_count = size_ref[0] * size_ref[1]
    #
    # g1 = 0.
    # g2 = 0.
    # r1 = 0.
    # b1 = 0.
    #
    # for i in range(size_ref[0]):
    #     for j in range(size_ref[1]):
    #         if (i % 2) == 0 and (j % 2) == 0: g1 = g1 + image_ref[i, j]
    #         if (i % 2) == 1 and (j % 2) == 1: g2 = g2 + image_ref[i, j]
    #         if (i % 2) == 0 and (j % 2) == 1: r1 = r1 + image_ref[i, j]
    #         if (i % 2) == 1 and (j % 2) == 0: b1 = b1 + image_ref[i, j]
    #
    # g1_avg = g1 / pix_count * 4
    # g2_avg = g2 / pix_count * 4
    # r1_avg = r1 / pix_count * 4
    # b1_avg = b1 / pix_count * 4
    #
    # raz_g1 = 1
    # raz_g2 = g1_avg/g2_avg
    # raz_r1 = g1_avg/r1_avg
    # raz_b1 = g1_avg/b1_avg
    #
    # avg = (g1+g2+r1+b1)/pix_count
    #
    # print(g1_avg, g2_avg, r1_avg, b1_avg, avg)

    raz_g1 = 1
    raz_g2 = 1.0245196396115988
    raz_r1 = 1.0131841989689434
    raz_b1 = 1.0517113199247086

    print('razmerje:', raz_g1, raz_g2, raz_r1, raz_b1)

    # in_path = 'd:\Projekti\Satelit\CO\Razpis\_POSNETKI\Peking_PAN\\'
    # in_pan_ref_file = 'NHDPfoc_swp6_4D_py.tif'
    # in_path = 'd:\Projekti\Satelit\CO\Razpis\Flat field images_new2020\\20201028 Vignetting\\flatfield\\'
    # in_pan_ref_file = 'NHDPflat_3D_py.tif'

    # in_path = 'd:\Projekti\Satelit\CO\Razpis\_POSNETKI\Slo_PAN\_26_30\\'
    # in_pan_ref_file = [filename for filename in os.listdir(in_path) if filename.lower().startswith("nhd") and filename.lower().endswith("tif")]

    

    

    # print('image', i)
    in_ref=file
    inreffil = gdal.Open(in_ref)
    image_ref = inreffil.ReadAsArray()
    size_ref = image_ref.shape
    # pix_count = size_ref[0] * size_ref[1]
    # pix_count = np.count_nonzero(image_ref)
    # pix_count = 3664*650

    # g1 = 0.
    # g2 = 0.
    # r1 = 0.
    # b1 = 0.
    #
    # for i in range(size_ref[0]):
    #     for j in range(size_ref[1]):
    #         if (i % 2) == 0 and (j % 2) == 0: g1 = g1 + image_ref[i, j]
    #         if (i % 2) == 1 and (j % 2) == 1: g2 = g2 + image_ref[i, j]
    #         if (i % 2) == 0 and (j % 2) == 1: r1 = r1 + image_ref[i, j]
    #         if (i % 2) == 1 and (j % 2) == 0: b1 = b1 + image_ref[i, j]
    #
    # g1_avg = g1 / pix_count * 4
    # g2_avg = g2 / pix_count * 4
    # r1_avg = r1 / pix_count * 4
    # b1_avg = b1 / pix_count * 4
    #
    # avg = (g1 + g2 + r1 + b1) / pix_count
    #
    # print(g1_avg, g2_avg, r1_avg, b1_avg, avg)

    # popravek
    im_p_pop = np.zeros((size_ref[0], size_ref[1]), np.uint16)


    for i in range(size_ref[0]):
        for j in range(size_ref[1]):
            if (i % 2) == 0 and (j % 2) == 0 and image_ref[i, j] != 0: im_p_pop[i, j] = image_ref[i, j] * raz_g1
            if (i % 2) == 1 and (j % 2) == 1 and image_ref[i, j] != 0: im_p_pop[i, j] = image_ref[i, j] * raz_g2
            if (i % 2) == 0 and (j % 2) == 1 and image_ref[i, j] != 0: im_p_pop[i, j] = image_ref[i, j] * raz_r1
            if (i % 2) == 1 and (j % 2) == 0 and image_ref[i, j] != 0: im_p_pop[i, j] = image_ref[i, j] * raz_b1
    
    _,_,_,_,P=return_flatfield_set_path(2)
    P_flat=gdal_array.LoadFile(P)
    
    # im_p_pop=simple_flatfield_corr(P_flat, im_p_pop, 2, 1)    
    
    # outout
    
    im_p_pop=BLUE_simple_flatfield_corr(P_flat, im_p_pop)
    
    out=os.path.abspath(file)+"/corr/"+os.path.basename(file)[:-4] + "_pop_flat_corr.tif"

    
    # out = in_ref[:-4] + "_pop_flat_corr.tif"

    driver = gdal.GetDriverByName('GTiff')

    # outRaster = driver.Create(out, size[1], size[0], 3, gdal.GDT_UInt16)
    outRaster = driver.Create(out, size_ref[1], size_ref[0], 1, gdal.GDT_UInt16)

    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(im_p_pop)
    outband.FlushCache()
    
    
    
def single_channel_stacking(tifs):
    """
single_channel_stacking

    Stacking and averaging multiple tifs around a template tif. Output image 
    is the same size as the template image

Parameters: tifs:list(strigs)
                filepath to the tif files
            
            
Returns:    None

    Saves the stacked and averaged tif at the same location as input
    
    """
    template_ID=int(len(tifs)/2)
        
    template_raster=gdal_array.LoadFile(tifs[template_ID-1])
    avg_raster=np.zeros_like(template_raster)
    avg_raster=avg_raster+1
    new_raster=np.copy(template_raster)
    # ones=np.full(template_raster.shape, 1)
    for i, tif in enumerate(tifs, start=1):
        if i==template_ID: 
            continue
    
        tif_raster=gdal_array.LoadFile(tif)
        # tif_raster=cut_transformed_array_borders(tif_raster)
        result=ird.similarity(template_raster,tif_raster , numiter=1, order=1)
        img_transformed= ird.transform_img(tif_raster, scale=result['scale'], angle=result['angle'], tvec=result['tvec'], mode='constant', bgval=0, order=2)
        
        img_transformed=cut_transformed_array_borders(img_transformed)
        
        # ones_transformed=ird.transform_img(ones, scale=result['scale'], angle=result['angle'], tvec=result['tvec'], mode='constant', bgval=0, order=1)
        ones_transformed=np.zeros_like(template_raster)
        ones_transformed[np.where(img_transformed>0)]=1
        print(ones_transformed)
        
        print(np.mean(ones_transformed), np.max(ones_transformed), np.min(ones_transformed))
        print(ones_transformed[np.where(ones_transformed>0)])
        print(np.min(ones_transformed[np.where(ones_transformed>0)]))
        print(np.max(ones_transformed[np.where(ones_transformed>0)]))

        plt.imshow(ones_transformed)
        plt.show()
        plt.close()
        
        # ones_transformed=cut_transformed_array_borders(ones_transformed)
        
        avg_raster=avg_raster+ones_transformed
        # ird.imshow(template_raster, tif_raster, img_transformed)
        
        new_raster=new_raster+img_transformed
        
    # new_raster=new_raster+template_raster   
    # new_raster=new_raster/len(tifs)

    gtz=np.where(avg_raster>0)
    

    

    
    
    plt.imshow(new_raster)
    plt.show()
    plt.close()
    # gdal_array.SaveArray(new_raster, tifs[0][:-4]+"_not_abvertaghe_stacked_.tiff")
    new_raster[gtz]=new_raster[gtz]/avg_raster[gtz]    
    gdal_array.SaveArray(new_raster, tifs[0][:-4]+"_stacked_.tiff")
    plt.imshow(new_raster)
    plt.savefig("test.tif", dpi=800)
    plt.show()
    plt.close()

    def discrete_cmap(N, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
    
        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:
    
        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)

    cmap=discrete_cmap(int(avg_raster.max())+1, base_cmap="ocean")    
    
    norm=mpl.colors.BoundaryNorm(np.arange(-0.5,int(avg_raster.max()+1)), cmap.N)
    fig=plt.figure()
    fig.set_size_inches(5,4)
    ax=fig.add_subplot(111)
    data=ax.matshow(avg_raster, cmap=cmap, norm=norm)
    fig.colorbar(data, ticks=np.linspace(0,int(avg_raster.max()),int(avg_raster.max()+1)), drawedges=True)

    plt.show()
    plt.close()


    # gdal_array.SaveArray(new_raster, tifs[0][:-4]+"_stacked_.tiff")
    
    
def single_channel_stacking_unlimited(tifs):
    """
single_channel_stacking_unlimited

    Stacking and averaging multiple tifs. The tifs are stacked in the 
    consecutive order. The output image is bigger than the single input image.
    
    

Parameters: tifs:list(strigs)
                filepath to the tif files
            
            
Returns:    None

    Saves the stacked and averaged tif at the same location as input
    
    """
    results=[]
    
    for i  in range(len(tifs)-1):
        r1=gdal_array.LoadFile(tifs[i])
        r2=gdal_array.LoadFile(tifs[i+1])
        print(tifs[i])
        print(tifs[i+1])
        result=ird.similarity(r1,r2 , numiter=1, order=1)
        print(result['tvec'])
        print(result['scale'])
        print(result['angle'])
        results.append(result)
        
    
        print(i)
    
    x0y0=(0,0)
    x_max_y_max=(r1.shape[1], r1.shape[0])
    cords=np.array([[x0y0[0], x_max_y_max[0],x0y0[1], x_max_y_max[1] ]])
    plt.scatter((cords[0,0],    cords[0, 1]), (cords[0,2],    cords[0, 3]))

    

    for i in range(len(tifs)-1):
        
        print(i)
        scale=0
        tvec_x=0
        tvec_y=0
        angle=0
        x0y0=(0,0)
        x_max_y_max=(r1.shape[1], r1.shape[0])
        
        for j in range(i+1):
            print(j)
            result=results[j]
            scale=result['scale']
            tvec_x=tvec_x+result['tvec'][1]
            tvec_y=tvec_y+result['tvec'][0]
            angle=angle+result['angle']
        M=Affine.translation(tvec_x,tvec_y )*Affine.scale(scale)*Affine.rotation(angle)
        print(M)
        x0y0=M*x0y0
        x_max_y_max=M*x_max_y_max
        
        cords=np.append(cords, [[x0y0[0], x_max_y_max[0],x0y0[1], x_max_y_max[1]]], axis=0)
        print(x0y0)
        print(x_max_y_max)
        
        plt.scatter((cords[i+1,0],    cords[i+1, 1]), (cords[i+1,2],    cords[i+1, 3]))
        
        
    xmin=np.min(cords[:,0:2])
    xmax=np.max(cords[:,0:2])
    ymin=np.min(cords[:,2:])
    ymax=np.max(cords[:,2:])
    
    print(cords)
    cords[:,0:2]=cords[:,0:2]-xmin
    cords[:,2:]=cords[:,2:]-ymin
    
    print(xmin,xmax, ymin,ymax)
    print(cords)
    
    
    
    final_array_shape=(int(np.abs(ymin-ymax)), int(np.abs(xmin-xmax)))
    print(final_array_shape)
    
    raster=np.zeros(final_array_shape)
    avg_raster=np.zeros(final_array_shape)
    
    # Mzero=Affine.translation(int(cords[0,0])+1, int(cords[0,2])+1)
    tif_raster=gdal_array.LoadFile(tifs[0])
    ones_raster=np.full_like(tif_raster, 1)    # ones_raster=np.full(tif_raster.shape, 1)
    
    pad_raster=np.zeros_like(raster)
    pad_raster[0:tif_raster.shape[0],0:tif_raster.shape[1]]=tif_raster
    pad_ones_raster=np.zeros_like(raster)
    pad_ones_raster[0:tif_raster.shape[0],0:tif_raster.shape[1]]=ones_raster
    
    
    pad_raster=ird.transform_img(pad_raster,tvec=(int(cords[0,2]),int(cords[0,0])), bgval=0)
    pad_raster=cut_transformed_array_borders(pad_raster)
    pad_ones_raster=ird.transform_img(pad_ones_raster,tvec=(int(cords[0,2]),int(cords[0,0])), bgval=0)
    pad_ones_raster=cut_transformed_array_borders(pad_ones_raster)
    # ones_raster=ird.transform_img(pad_raster,tvec=(int(cords[0,2])+1,int(cords[0,0])+1))
    # where_ones=np.where(pad_raster>0)
    # ones_raster[where_ones]=1
    raster=raster+pad_raster
    avg_raster=avg_raster+pad_ones_raster
    
    # for i in range(zero_raster.shape[0]):
    #     for j in range(zero_raster.shape[1]):
    #         xy=(j,i)
    #         new_xy=Mzero*xy
    #         new_xy=[new_xy[0], new_xy[1]]
    #         new_xy[0]=int(new_xy[0])
    #         new_xy[1]=int(new_xy[1])
            
    #         raster[new_xy[1], new_xy[0]]=zero_raster[i,j]
    #         avg_raster[new_xy[1], new_xy[0]]=avg_raster[new_xy[1], new_xy[0]]+1
    
    for k,tif in enumerate(tifs, start=1):
        print(tif)
        if k==1:
            continue
        scale=1
        tvec_x=0
        tvec_y=0
        angle=0
        
        for r in range(k-1):
            result=results[r]
            scale=scale*result['scale']
            tvec_x=tvec_x+result['tvec'][1]
            tvec_y=tvec_y+result['tvec'][0]
            angle=angle+result['angle']
        tvec_x=tvec_x-xmin
        tvec_y=tvec_y-ymin
        M=Affine.translation(tvec_x,tvec_y )*Affine.scale(scale)*Affine.rotation(angle)
        print(M)
        x0y0=M*x0y0
        x_max_y_max=M*x_max_y_max
        
        tif_raster=gdal_array.LoadFile(tif)
        
        
        # for i in tqdm(range(tif_raster.shape[0]), desc="transforming: "+tif):
        #     for j in range(tif_raster.shape[1]):
        #         xy=(j,i)
        #         new_xy=M*xy
        #         new_xy=[new_xy[0], new_xy[1]]
        #         new_xy[0]=int(new_xy[0])
        #         new_xy[1]=int(new_xy[1])
                
        #         raster[new_xy[1], new_xy[0]]=raster[new_xy[1], new_xy[0]]+tif_raster[i,j]
        #         avg_raster[new_xy[1], new_xy[0]]=avg_raster[new_xy[1], new_xy[0]]+1
        
        
        pad_raster=np.zeros_like(raster)
        pad_raster[0:tif_raster.shape[0],0:tif_raster.shape[1]]=tif_raster
        ones_raster=np.full_like(tif_raster, 1)
        pad_ones_raster=np.zeros_like(raster)
        pad_ones_raster[0:tif_raster.shape[0],0:tif_raster.shape[1]]=ones_raster
    
        
        
        pad_raster=ird.transform_img(pad_raster,scale=scale, angle=angle, tvec=(tvec_y, tvec_x), mode='constant', bgval=0)
        pad_ones_raster=ird.transform_img(pad_ones_raster,scale=scale, angle=angle, tvec=(tvec_y, tvec_x), mode='constant', bgval=0)
        # ones_raster=ird.transform_img(pad_raster,tvec=(int(cords[0,2])+1,int(cords[0,0])+1))
        # where_ones=np.where(pad_raster>0)
        # ones_raster[where_ones]=1
        raster=raster+pad_raster
        # avg_raster=avg_raster+ones_raster
        avg_raster=avg_raster+pad_ones_raster
    
    # left_border=xmin
    # upper_border=ymax    
    # print(raster.shape)


    





    plt.show()
    plt.close()  
    
    gtz=np.where(avg_raster>0)
    
    raster[gtz]=raster[gtz]/avg_raster[gtz]
    basename=os.path.basename(tif)
    gdal_array.SaveArray(raster, os.path.dirname(os.path.abspath(tif))+"/stacked/"+basename[:-16]+"_py_corr_stackeg_big_.tif")
    
    def discrete_cmap(N, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
    
        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:
    
        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)

    cmap=discrete_cmap(int(avg_raster.max())+1, base_cmap="ocean")    
    
    norm=mpl.colors.BoundaryNorm(np.arange(-0.5,int(avg_raster.max()+1)), cmap.N)
    fig=plt.figure()
    fig.set_size_inches(15,10)
    ax=fig.add_subplot(111)
    data=ax.matshow(avg_raster, cmap=cmap, norm=norm)
    fig.colorbar(data, ticks=np.linspace(0,int(avg_raster.max()),int(avg_raster.max()+1)), drawedges=True)

    plt.show()
    plt.close()
    
    
    plt.imshow(avg_raster)
    plt.show()
    plt.close()
    
    plt.imshow(raster)
    plt.show()
    plt.close()
    
    
def rgb_stacking(tifs):
    template_ID=int(len(tifs)/2)
        
    template_raster=gdal_array.LoadFile(tifs[template_ID-1][:,:,2])
    avg_raster=np.zeros_like(template_raster)
    avg_raster=avg_raster+1
    
    new_raster=np.copy(template_raster)
    
    
    # ones=np.full(template_raster.shape, 1)
    
    
    
    
    for i, tif in enumerate(tifs, start=1):
        if i==template_ID: 
            continue
    
        tif_raster=gdal_array.LoadFile(tif)
        # tif_raster=cut_transformed_array_borders(tif_raster)
        result=ird.similarity(template_raster,tif_raster[:,:,2] , numiter=1, order=1)
        img_transformed= ird.transform_img(tif_raster, scale=result['scale'], angle=result['angle'], tvec=result['tvec'], mode='constant', bgval=0, order=1)
        
        img_transformed[:,:,0]=cut_transformed_array_borders(img_transformed[:,:,0])
        img_transformed[:,:,1]=cut_transformed_array_borders(img_transformed[:,:,1])
        img_transformed[:,:,2]=cut_transformed_array_borders(img_transformed[:,:,2])

        
        # ones_transformed=ird.transform_img(ones, scale=result['scale'], angle=result['angle'], tvec=result['tvec'], mode='constant', bgval=0, order=1)
        ones_transformed=np.zeros_like(template_raster)
        ones_transformed[np.where(img_transformed>0)]=1
        print(ones_transformed)
        
        print(np.mean(ones_transformed), np.max(ones_transformed), np.min(ones_transformed))
        print(ones_transformed[np.where(ones_transformed>0)])
        print(np.min(ones_transformed[np.where(ones_transformed>0)]))
        print(np.max(ones_transformed[np.where(ones_transformed>0)]))

        plt.imshow(ones_transformed)
        plt.show()
        plt.close()
        
        # ones_transformed=cut_transformed_array_borders(ones_transformed)
        
        avg_raster=avg_raster+ones_transformed
        # ird.imshow(template_raster, tif_raster, img_transformed)
        
        new_raster=new_raster+img_transformed
        
    # new_raster=new_raster+template_raster   
    # new_raster=new_raster/len(tifs)

    gtz=np.where(avg_raster>0)
    

    

    
    
    plt.imshow(new_raster)
    plt.show()
    plt.close()
    gdal_array.SaveArray(new_raster, tifs[0][:-4]+"_not_abvertaghe_stacked_.tiff")
    new_raster[:,:,0][gtz]=new_raster[:,:,0][gtz]/avg_raster[gtz]
    new_raster[:,:,1][gtz]=new_raster[:,:,1][gtz]/avg_raster[gtz]    
    new_raster[:,:,2][gtz]=new_raster[:,:,2][gtz]/avg_raster[gtz]    
    
    gdal_array.SaveArray(new_raster, tifs[0][:-4]+"_YES_abvertaghe_stacked_.tiff")
    
    plt.imshow(new_raster)
    plt.savefig("test.tif", dpi=800)
    plt.show()
    plt.close()

    def discrete_cmap(N, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
    
        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:
    
        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)

    cmap=discrete_cmap(int(avg_raster.max())+1, base_cmap="ocean")    
    
    norm=mpl.colors.BoundaryNorm(np.arange(-0.5,int(avg_raster.max()+1)), cmap.N)
    fig=plt.figure()
    fig.set_size_inches(5,4)
    ax=fig.add_subplot(111)
    data=ax.matshow(avg_raster, cmap=cmap, norm=norm)
    fig.colorbar(data, ticks=np.linspace(0,int(avg_raster.max()),int(avg_raster.max()+1)), drawedges=True)

    plt.show()
    plt.close()
    
    
def single_image_band_match(tif):
    """
single_image_band_match

    Aligning seperate channels in an single image.

Parameters: tifs:list(strigs)
                filepath to the tif files
            
            
Returns:    None

    Saves the stacked tif at the same location as input
    
    """
    tif_raster=gdal_array.LoadFile(tif)
    
    file_path=tif[:-4]+"matched.tif"
    
    R=tif_raster[0]
    G=tif_raster[1]
    B=tif_raster[2]
    
    print(R.shape)
    result=ird.similarity(G,R , numiter=1, order=1)
    R= ird.transform_img(R, scale=result['scale'], angle=result['angle'], tvec=result['tvec'], mode='constant', bgval=0, order=1)
    # print(result)
    print(R.shape)
    
    print(B.shape)
    result=ird.similarity(G,B , numiter=1, order=1)
    B= ird.transform_img(B, scale=result['scale'], angle=result['angle'], tvec=result['tvec'], mode='constant', bgval=0, order=1)
    # print(result)
    print(B.shape)
    driver=osgeo.gdal.GetDriverByName("GTiff")
    options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
    
    dataset=driver.Create(file_path, R.shape[1],R.shape[0],  3, osgeo.gdal.GDT_UInt16, options)        
    dataset.GetRasterBand(1).WriteArray(R)
    dataset.GetRasterBand(2).WriteArray(G)
    dataset.GetRasterBand(3).WriteArray(B)
    
    
def rgb_to_hsv(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v

def hsv_to_rgb(h, s, v):
    i = math.floor(h*6)
    f = h*6 - i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1-(1-f)*s)

    r, g, b = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][int(i%6)]

    return r, g, b



def equalizeHist_HSV(raster):
    raster=np.transpose(raster,(1,2,0))
    print(raster.shape)
    raster = cv2.cvtColor(raster, cv2.COLOR_RGB2BGR)
    print(raster.shape)
    raster = cv2.cvtColor(raster, cv2.COLOR_BGR2HSV)
    print(raster.shape)
    # raster[:,:,0]=cv2.equalizeHist(raster[:,:,0])
    # raster[:,:,1]=cv2.equalizeHist(raster[:,:,1])
    raster[:,:,2]=cv2.equalizeHist(raster[:,:,2])
    print(raster.shape)
    raster = cv2.cvtColor(raster, cv2.COLOR_HSV2RGB)
    print(raster.shape)
    return raster
    

def create_prewiev_jpg(tif_file=None, tif_raster=None):
    if tif_raster is None:
        raster=gdal_array.LoadFile(tif_file)
    if tif_file is None:
        raster=tif_raster
    
    
    if raster.ndim==2:
        raster=(raster/np.max(raster))*255
        raster=raster.astype(np.uint8)
        raster=cv2.equalizeHist(raster)
        
    else:
        print(raster.shape)
        raster=(raster/np.max(raster))*90
        print(raster.shape)
        raster=raster.astype(np.uint8)
        print(raster.shape)
       
        # raster=equalizeHist_HSV(raster)
        
        raster[0]=(raster[0]/raster[0].max())*255
        raster[1]=(raster[1]/raster[1].max())*255
        raster[2]=(raster[2]/raster[2].max())*200
        
        # raster[0]=cv2.equalizeHist(raster[0])
        # raster[1]=cv2.equalizeHist(raster[1])
        # raster[2]=cv2.equalizeHist(raster[2])
        

        
        raster=np.transpose(raster,(1,2,0))
    plt.imshow(raster)
    plt.colorbar()
    plt.show()
    plt.close()
    raster = cv2.cvtColor(raster, cv2.COLOR_RGB2BGR)
    
    if tif_file!=None:
        cv2.imwrite(tif_file[:-4]+"_preview.jpg", raster)
    


def single_channel_tifs(tifs_location,channel):
    """
single_channel_tifs

    Searches 

Parameters: tifs_location:str
                Location of the folder containing tif files
            channel:str
                Channel of the selected images (R, G, B, N, P)
            
            
Returns:    list(strings)

    Return list of strings, containing filepaths to tifs with "channel" in the
    4. place in the file name. Current NemoHD nomenclature.
    
    """
    tifs=glob.glob(tifs_location+"*.tif")
    
    channel_tifs=[]
    for tif in tifs:
        if os.path.basename(tif)[3]==channel:
            channel_tifs.append(tif)
    print(str(len(channel_tifs))+" "+channel+"-channel tifs found")
    return channel_tifs