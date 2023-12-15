# Super-resolution

In this file we will walk you through the codes and the work flow used in the whole super-resolution process.

GenerateLowResolution:

This function will read images from the selected folder and the return low resolution images;
The function will take the following parameters:
  - folder: Folder (folder's path) containing the high resolution images - string
  - qty: The quantity of low resolution images that will be generated - integer
  - rf: The rescaling factor that will be applied to the images in order to chnage its size - integer
  - k: Kernel's size tha will be used to apply the filter - tuple -> (3,3)
  - theta: Angle's variation range - tuple -> (0,1)
  - tx: Range of translation on the horizontal axis - tuple -> (-10,10)
  - ty: Range of translation on vertical axis - tuple -> (-10,10)

in the beginning, the funtion will start list all the files contained inside the folder. Right after that, it will read all of them 
returning an image or a "none" value. If the reading is an actual image, a new loop will begin creating the folder for the low resolution images generated
and checking if this folder already exists. Then the read image will be shrinked, blurred, translated, and rotated. In the end, it will be saved with a hexadecimal 6 digits number as its name
and the data and parameters used in the whole proces will be written in a csv file. In case it doesn't exist, it will be created.

In the following steps the generated images will be used in the registration, alignment and transformation process, 
these process make up the super resoltion process

DemonsRegistration:

this function will use the demons registration process from Simple ITK 
, and right after that it will execute the alignment of the images and return the transform function.
it will only take the folder (its path) containing the high resolution images as a parameter in code.
- folder = 'images'
