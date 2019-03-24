# Image-stacking
Image stacking for lower SNR

## Dropbox source frames and results:
[frames and results](https://www.dropbox.com/sh/9lmp1ietl2mlydv/AABHkiXNAkWDFf-j9yzsKf1na?dl=0)

## Used libraries:
- [imreg_dft](https://github.com/matejak/imreg_dft)

  Image registration using discrete Fourier transform. 

- [Gdal](https://www.gdal.org/)

  Geospatial Data Abstraction Library. In this case used for reading and writing multichannel images.
  
- [glob](https://docs.python.org/3/library/glob.html)

  Glob is used for *tif-file listing in the source directory.
  
## Algoritem description:
- All tif images are listed in the choosen folder
- The Images are saved in four 3D numpy array (one array per color channel, RGB_IR)

### Input-data:
- Chosem template image (template_id) --> The one image all other are aligned to
- Chosem channel for image registration (channel_id)

#### imreg_dft parameters:
- number of iterations (calculation of scale and rotation)
- order of approximation

The image registration is calculated only on one chosen channel. All other channels are transformed based on the resulting data (translation, rotation, scale).

### Multiprocessing:
Pythons build in librarie multiprocessing is used. 
- multiprocessing.Process
- multiprocessing.RawArray (simple 1D array that diferrent processes can share)

#### part_channel_reg
Every process (4, one per CPU core) is running the imreg_dft.similarity function on part of the selected color channel. The resulting aligned images (after the transformation) are transformed into a 1D shape (numpy.reshape) and are stored into the shared arrays (one per channel).
#### single_channel_average
The shared arrays are passed to the function. Four processes are run. One process calculates the average image for one channel.

