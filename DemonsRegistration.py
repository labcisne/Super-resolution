import SimpleITK as sitk
import os
import numpy as np
from skimage.transform import resize
import json

def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(
            original_size, original_spacing, new_size
        )
    ]
    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def multiscale_demons(
    registration_algorithm,
    fixed_image,
    moving_image,
    initial_transform=None,
    shrink_factors=None,
    smoothing_sigmas=None,
):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                        are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(
            list(zip(shrink_factors, smoothing_sigmas))
        ):
            fixed_images.append(
                smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma)
            )
            moving_images.append(
                smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma)
            )

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(
            initial_transform,
            sitk.sitkVectorFloat64,
            fixed_images[-1].GetSize(),
            fixed_images[-1].GetOrigin(),
            fixed_images[-1].GetSpacing(),
            fixed_images[-1].GetDirection(),
        )
    else:
        initial_displacement_field = sitk.Image(
            fixed_images[-1].GetWidth(),
            fixed_images[-1].GetHeight(),
            #fixed_images[-1].GetDepth(),
            sitk.sitkVectorFloat64,
        )
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    initial_displacement_field = registration_algorithm.Execute(
        fixed_images[-1], moving_images[-1], initial_displacement_field
    )
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in reversed(
        list(zip(fixed_images[0:-1], moving_images[0:-1]))
    ):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(
            f_image, m_image, initial_displacement_field
        )
    return sitk.DisplacementFieldTransform(initial_displacement_field)

# Define a simple callback which allows us to monitor registration progress.
def iteration_callback(filter):
    print(
        "\r{0}: {1:.2f}".format(filter.GetElapsedIterations(), filter.GetMetric()),
        end="",
    )

folder = 'imagens'

high_res_images= []

for file in os.listdir(folder):
    if os.path.isfile(os.path.join(folder,file)):
        high_res_images.append(file)

for img in high_res_images:

    low_res_images= []
    folder_LR = f'LR - {img}'
    folder_LR_path = os.path.join(folder,folder_LR)
    for lr_file in os.listdir(folder_LR_path):
        if os.path.isfile(os.path.join(folder_LR_path,lr_file)):
            low_res_images.append(lr_file)
    
    path_fix = os.path.join(folder_LR_path, low_res_images[0])
    images= []
    images.append(sitk.ReadImage(path_fix, sitk.sitkFloat32))

    matrices_list = []
    for file_mov in range(1,len(low_res_images)):
        path_mov=  os.path.join(folder_LR_path, low_res_images[file_mov])
        images.append(sitk.ReadImage(path_mov, sitk.sitkFloat32))
    
        fixed_index = 0
        moving_index = file_mov

        fixed_image = images[fixed_index]
        moving_image = images[moving_index]


        # Select a Demons filter and configure it.
        demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(20)
        # Regularization (update field - viscous, total field - elastic).
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(2.0)

        # Add our simple callback to the registration filter.
        demons_filter.AddCommand(
            sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter)
        )

        # # Run the registration.
        tx = multiscale_demons(
            registration_algorithm=demons_filter,
            fixed_image=fixed_image,
            moving_image=moving_image,
            shrink_factors=[1, 1],
            smoothing_sigmas=[8, 4],
        )

        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            tx,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
        
        matx = sitk.GetArrayFromImage(moving_resampled)
        print(matx)
        matrices_list.append(matx)
    #############################################################3
    SR = []
    confidenceweight = {}
    optimParams = {}
    model = {}
    coarseToFineScaleFactors = []

    if len(SR) == 0:
        tensor = np.stack(matrices_list,axis=-1)
        print(tensor.shape)
        sr_med = np.zeros((tensor.shape[0],tensor.shape[1]))
        for y in range(tensor.shape[0]):
            for x in range(tensor.shape[1]):
                a = np.array(tensor[y,x,:])
                for value in a:
                    if value>0:
                        np.delete(a,np.where(a==0))
                sr_med[y,x] = np.median(a)
        print(sr_med)
    
      #Setting Scale
        np.array(sr_med)
        scale = 1
        img_rsz = resize(sr_med,(sr_med.shape[0],sr_med.shape[1]),anti_aliasing= True )
        print(img_rsz)
        SR = sr_med
