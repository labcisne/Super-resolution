import SimpleITK as sitk
import os
import numpy as np
from skimage.transform import resize
import json
import numpy as np
from skimage.transform import resize
import subprocess
import scipy.io as sio
from scipy.sparse import csc_matrix, coo_array
# from scipy import sparse
import json
import imresize
from scipy.signal import medfilt2d
from scipy.optimize import fmin_cg

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

def imageToVector(img):
    return img.flatten('F')[:,np.newaxis]

def get_residual(SR, LR, W, photometric_params=None):
    # GETRESIDUAL Get residual error for low-resolution and super-resolved
    # data.
    # GETRESIDUAL computes the residual error caused by a super-resolved
    # image with the associated low-resolution frames and the model
    # parameters (system matrix and photometric parameters).

    if photometric_params is None:
        sSR = csc_matrix(SR)
        print(LR.shape)
        print(W.shape)
        print(sSR.shape)
        print(type(W))
        print(type(sSR))
        print('não fez')
        r = LR - np.multiply(W,sSR)
        print('fez')
    else:
        num_frames = photometric_params.mult.shape[2]
        num_lr_pixel = len(LR) // num_frames

        if np.ndim(photometric_params.mult) == 2:  # Check if 'mult' is a vector
            bm = np.zeros_like(LR)
            ba = np.zeros_like(LR)
            for k in range(num_frames):
                start_idx = (k * num_lr_pixel)
                end_idx = ((k + 1) * num_lr_pixel)
                bm[start_idx:end_idx] = np.tile(photometric_params.mult[:,:,k], (num_lr_pixel, 1)).flatten()
                ba[start_idx:end_idx] = np.tile(photometric_params.mult[:,:,k], (num_lr_pixel, 1)).flatten()
        else:
            bm = np.zeros_like(LR)
            ba = np.zeros_like(LR)
            for k in range(num_frames):
                start_idx = (k * num_lr_pixel)
                end_idx = ((k + 1) * num_lr_pixel)
                bm[start_idx:end_idx] = photometric_params.mult[:,:,k].flatten()
                ba[start_idx:end_idx] = photometric_params.add[:,:,k].flatten()

        r = LR - bm * np.dot(W, SR) - ba

    return r

def estimate_observation_confidence_weights(r, weights=None, scale_parameter=None):
    # Estimate observation confidence weights
    weights_vec = []
    r_vec = []
    r_max = 0.02  # Threshold for median residual to detect outliers
    # r_max = 0.1  # Alternative threshold (commented out in MATLAB)
    print(len(r))
    print(len(weights))

    for k in range(len(r)):
        r_vec.extend(r[k])
        if np.abs(np.median(r[k])).any() < r_max:
            weights_bias_k = 1
        else:
            weights_bias_k = 0
        weights_vec.extend([w * weights_bias_k for w in weights[k]])

    if scale_parameter is None:
        if any(w > 0 for w in weights_vec):
            # Use adaptive scale parameter estimation
            scale_parameter = get_adaptive_scale_parameter(np.array(r_vec)[np.array(weights_vec) > 0],
                                                           np.array(weights_vec)[np.array(weights_vec) > 0])
        else:
            scale_parameter = 1

    for k in range(len(r)):
        # Estimate local confidence weights for current frame
        c = 2
        weights_local = 1 / np.abs(r[k])
        weights_local[np.abs(r[k]) < c * scale_parameter] = 1 / (c * scale_parameter)
        weights_local = c * scale_parameter * weights_local

        # Assemble confidence weights from bias weights and local weights
        if any(weights_bias_k > 0 for weights_bias_k in weights_vec):
            weights[k] = [weights_bias_k * weights_local_i for weights_bias_k, weights_local_i in zip(weights_vec, weights_local)]
        else:
            weights[k] = [1 * weights_local_i for weights_local_i in weights_local]

    return weights, scale_parameter

def get_adaptive_scale_parameter(r, weights):
    # Nested function to compute adaptive scale parameter
    weighted_median_diff = np.abs(r - weighted_median(r, weights))
    scale_parameter = 1.4826 * weighted_median(weighted_median_diff[weights > 0], weights[weights > 0])
    return scale_parameter

def weighted_median(values, weights):
    # Function to compute weighted median
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(sorted_weights)
    
    median_index = np.searchsorted(cumulative_weights, total_weight / 2.0)
    return sorted_values[median_index]

def get_btv_transformed_image(X, P, alpha0):
    # Pad image at the border to perform shift operations
    Xpad = np.pad(X, ((P, P), (P, P)), mode='symmetric')
    
    # Consider shifts in the interval [-P, +P]
    btv_transformed_image = []
    rows, cols = X.shape
    for l in range(-P, P + 1):
        for m in range(-P, P + 1):
            if l != 0 or m != 0:
                # Shift by l and m pixels
                Xshift = Xpad[(P + l):(P + l + rows), (P + m):(P + m + cols)]
                
                # Apply BTV transformation
                transformed_patch = alpha0**(np.abs(l) + np.abs(m)) * (Xshift - X)
                
                # Apply median filtering
                filtered_patch = medfilt2d(transformed_patch, kernel_size=3)
                
                # Convert the filtered patch to a vector and append to btv_transformed_image
                btv_transformed_image.append(filtered_patch.flatten())
    
    btv_transformed_image = np.array(btv_transformed_image)
    
    return btv_transformed_image

class Parametro:
    def __init__(self):
        self.P = None
        self.alpha = None
        self.photometricParams = None
        self.imagePrior = None
        self.confidence = None
        self.SR = None
        self.magFactor = None
        self.psfWidth = None
        self.motionParams = None
        self.priori = None
        
    
    def setMotionParams(self,numFrames,registration=False):
        if registration == False:
            motionParams = []
            for i in range(numFrames):
                motionParams.append(np.eye(3))
            self.motionParams = np.array(motionParams)
        else:
            pass # Registro simultâneo à reconstrução, não implementaremos isso

model = Parametro()
model.P = 2
model.alpha = 0.5
model.magFactor = 2
model.psfWidth = 0.4
model.magFactor = 8
model.motionParams = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
model.confidence = []

class imagePrior():
    def __init__(self):
        self.weight = None

model.imagePrior = imagePrior()
model.imagePrior.weight = 1 

class optimParams():
    def __init__(self):
        self.maxCVIter = None
        self.maxMMiter = None


SR = []
confidenceweight = {}
coarseToFineScaleFactors = [*range(min(1,model.magFactor), model.magFactor)]
Y= []
residualError = []
W_mat = []

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
        #print(matx)
        matrices_list.append(matx)
    #############################################################3
    SR = []

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
        #print(sr_med)
    
      #Setting Scale
        np.array(sr_med)
        scale = 1
        img_rsz = resize(sr_med,(sr_med.shape[0],sr_med.shape[1]),anti_aliasing= True )
        #print(img_rsz)
        SR = sr_med

    SR = imageToVector(SR)

    for frameIdx in range(0,tensor.shape[2]):
        model.confidence.append(np.ones((tensor.shape[0]*tensor.shape[1],1)))

    maxCVIter = optimParams.maxCVIter = 2
    maxMMiter = optimParams.maxMMiter = 2
    
    if model.imagePrior.weight is not None:
            useFixedRegularizationWeight = True
    else:
        useFixedRegularizationWeight = False

    for iter in range(0,optimParams.maxMMiter):
        W=[]
        residualError=[]
        if iter <= len(coarseToFineScaleFactors):
            model.magFactor = coarseToFineScaleFactors[iter]

            for frameIdx in range(0,tensor.shape[2]):
                params = {'imsize': (tensor.shape[0],tensor.shape[1]), 'magFactor': model.magFactor, 'psfWidth' :  model.psfWidth, 'motionParams':  model.motionParams }
                json_object = json.dumps(params, indent=8)
                with open("params.json", "w") as outfile:
                    outfile.write(json_object)
                matBin = "/home/labcisne/R2022a/bin/matlab" 
                # Necessario definir variavel folder se o .py nao estiver na mesma pasta do composeSM.m, CC folder = ""
                folder = "/home/labcisne/DocThais/DocThais/Projeto/Testes/SRPython"
                # folder = ""
                script = "testeComposeSystemMarix.m"

                cmd = []
                cmd.append(matBin)
                cmd.append("-sd")
                cmd.append(folder)
                cmd.append("-batch")
                cmd.append(f"run('{script}');")

                subprocess.run(cmd)
                W_mat = sio.loadmat('Wmx.mat')
                W.append(W_mat['W2'] )
                print('fez 1')
                        
            if iter>0:
                print('CHHEEEGGOOOUUUUUUUUUUUUUUUU"')
                SR= imresize.imresize(np.reshape(SR,(tensor.shape[0]*coarseToFineScaleFactors[iter-1],tensor.shape[1]*coarseToFineScaleFactors[iter-1]),'C'),
                        (coarseToFineScaleFactors[iter]/coarseToFineScaleFactors[iter-1]))
                SR = imageToVector(SR)
        print(f"Elementos lista W: {len(W)}")
        for frameIdx in range(0,tensor.shape[2]):
            print(f'Imagem número:{frameIdx}\n')
            Y.append(imageToVector(tensor[:,:,frameIdx]))
            #residualError[frameIdx] = get_residual(SR, Y[frameIdx],W[frameIdx], model.photometricParams)
            residualError.append(get_residual(SR, Y[frameIdx],W[frameIdx], model.photometricParams)) 
            print('fez 2')
        
        model.confidence, sigmaNoise = estimate_observation_confidence_weights(residualError, model.confidence)
        
        
        

        