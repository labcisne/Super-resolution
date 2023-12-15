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

DemonsRegistration:

this function will use the demons registration process from Simple ITK 
, and right after that it will execute the alignment of the images and return the transform function.
it will only take the folder (its path) containing the high resolution images as a parameter in code.
- folder = 'images'
