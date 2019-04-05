# Image-stacking
Image stacking for lower SNR

## Usage
### Running
python image_registration.py template_id channel_id averaging_mode file_name
#### example
python image_registration.py 3 cmd data
### The scripts takes 3 arguments:
1. template_id is automatically set --> int(number_of_files/2)
2. channel_id (integer) 0-number_of_channels
3. averaging mode (string) cmd,crp. cmd --> returns the full sizes image with partial to complete overlaying. crp --> returns a cropped image of the area where all filles are overlaying
4. file_name (string, optional), in addition to the channel_id, template_id, and averaging mode, file_name is added to the basename of the text file where the transformation results are saved. If non provided, the script doesnt write to txt.

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
Pythons build in library multiprocessing is used. 
- multiprocessing.Process
- multiprocessing.RawArray (simple 1D array that different processes can share)

#### part_channel_reg
Every process (4, one per CPU core) is running the imreg_dft.similarity function on part of the selected color channel. The resulting aligned images (after the transformation) are transformed into a 1D shape (numpy.reshape) and are stored into the shared arrays (one per channel).
#### single_channel_average
The shared arrays are passed to the function. Four processes are run. One process calculates the average image for one channel.

