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
