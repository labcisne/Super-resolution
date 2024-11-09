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
from scipy.optimize import minimize
from scipy.ndimage import shift


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

def getBTVTransformedImage(X, P, alpha0):
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

# def updateHighResolutionImage(SR, optimParams, model, y, W, Wt):
#     # Setup parameters for CG optimization.
#     cgOptions = {'gtol': optimParams.terminationTol,
#                  'maxiter': optimParams.maxSCGIter,
#                  'disp': False}
    
#     # Perform CG iterations to update the current estimate of the
#     # high-resolution image.
#     if SR.ndim == 1:
#         SR = SR[:, np.newaxis]  # Convert to column vector if SR is 1D

#     def imageObjectiveFunc(x, *args):
#         # Define your objective function here
#         # For example:
#         # return np.sum((y - model @ x)**2)
#         pass

#     def imageObjectiveFunc_grad(x, *args):
#         # Define your gradient of the objective function here
#         # For example:
#         # return -2 * model.T @ (y - model @ x)
#         pass

#     SR, _, _ = fmin_cg(imageObjectiveFunc, SR.flatten(), fprime=imageObjectiveFunc_grad,
#                        args=(model, y, W, Wt), **cgOptions)

#     numIters = _['nfev']  # Assuming 'nfev' gives the number of iterations
#     SR = SR.flatten()

#     return SR, numIters


def updateHighResolutionImage(SR, model, y, W, Wt, optimParams, imsize):
    print(f'Linha 290 y é do tipo {type(y)}')
    """
    Update the high-resolution image estimate using the Scaled Conjugate Gradient (SCG) method.
    
    Parameters:
        SR (np.ndarray): Initial estimate of the high-resolution image.
        model: Model related to the image reconstruction process.
        y: Observed data.
        W: Operator matrix for the image reconstruction process.
        Wt: Transpose of the operator matrix.
        optim_params: Dictionary containing optimization parameters (terminationTol, maxSCGIter).
    
    Returns:
        np.ndarray: Updated high-resolution image estimate.
        int: Number of iterations performed.
    """

    # def image_objective_func(SR_flat):
    #     # Placeholder for the actual objective function
    #     return np.sum((W @ SR_flat - y) ** 2)  # Example function, modify as needed

    def image_objective_func(SR, model, y, W):
        """
            Compute the objective function for the image super-resolution problem.
            
            Parameters:
                SR (np.ndarray): Current estimate of the high-resolution image.
                model: Model containing the prior function and its parameters.
                y: Observed data.
                W: List of operators applied to the estimate SR.
            
            Returns:
                float: Value of the objective function.
        """
        print(f'SR- {type(SR)}')
        print(f'model {type(model)}')
        print(f'y {type(y)}')
        print(f'W {type(W)}')
 
        # Ensure SR is a column vector
        if SR.ndim == 1:
            SR = SR.reshape(-1, 1)
        elif SR.shape[1] != 1:
            SR = SR.T
            
        # Evaluate the data fidelity term
        data_term = 0
        for k in range(len(y)):
            diff = y[k] - W[k] @ SR
            data_term += np.sum(model.confidence[k] * (diff ** 2))
        
        # Evaluate the image prior term for regularization
        prior_term = btv_prior_weighted(SR, imagePrior.weight,imsize)
        
        # Calculate the objective function
        f = data_term + imagePrior.weight * prior_term
        
        return f

    def image_objective_func_grad(SR_flat):
        # Placeholder for the actual gradient of the objective function
        return 2 * W.T @ (W @ SR_flat - y)  # Example gradient, modify as needed

    # Convert SR to a 1D array if it is not already
    SR_flat = SR.flatten()

    # # Optimization options
    # scgOptions = np.zeros((1, 18))
    # scgOptions[0,2]= optimParams.terminationTol
    # scgOptions[0,3]= optimParams.terminationTol
    # scgOptions[0,10]= optimParams.maxSCGIter
    # scgOptions[0,14]= optimParams.maxSCGIter
    
    # Optimization options
    options = {
        'gtol': optimParams.terminationTol,  # Gradient norm must be smaller than this value
        'maxiter': optimParams.maxSCGIter  # Maximum number of iterations
    }


    # Perform optimization
    # result = minimize(image_objective_func, SR_flat, method='CG', jac=image_objective_func_grad, options=scgOptions)
    print(f'Linha 371 y é tipo {type(y)}')
    result = minimize(image_objective_func(SR, model, y, W), SR_flat, method='CG', jac=image_objective_func_grad, options=options)


    # Extract the optimized high-resolution image and number of iterations
    SR_updated = result.x.reshape(SR.shape)
    num_iters = result.nit
    
    return SR_updated, num_iters

def isConverged(SR, SR_old, optimParams):
    converged = np.max(np.abs(SR_old - SR)) < optimParams.terminationTol
    return converged

def selectRegularizationWeight(SR, optimParams, model, y, W):
    # Initialize variables
    maxCVIter = optimParams.maxCVIter
    fractionCvTrainingObservations = optimParams.fractionCVTrainingObservations
    hyperparameterCVSearchRange = optimParams.hyperparameterCVSearchRange
    
    bestLambda = None
    SR_best = SR
    minValError = np.inf
    
    # # Split the set of given observations into training and validation subset.
    trainObservations = []
    y_train = []
    y_val = []
    W_train = []
    Wt_train = []
    W_val = []
    
    for k in range(len(y)):
        trainObservations_k = 1- np.random.rand(len(y[k])) > fractionCvTrainingObservations
        trainObservations.append(trainObservations_k)
        y_train.append(y[k][trainObservations_k])
        y_val.append(y[k][~trainObservations_k])
        W_train.append(W[k][trainObservations_k, :])
        Wt_train.append(W_train[k].T)
        W_val.append(W[k][~trainObservations_k, :])
        # print(f'train  {trainObservations}')  
        # print(f'y_train  {y_train}')
        # print(f'y_val  {y_val}')

    print(type(trainObservations))
    trainObservations=np.array(trainObservations)
    print(type(trainObservations))
    trainObservations_n=trainObservations.astype(int)

    # trainObservations = []
    # y_train = []  
    # y_val = []
    # W_train = []
    # Wt_train = []
    # W_val = []

    # fractionCvTrainingObservations = 0.8  # Defina o valor de fração como um exemplo

    # for k in range(len(y)):
    #     # Cria um array booleano com a mesma forma que y[k]
    #     trainObservations_k = 1 - np.random.rand(len(y[k])) > fractionCvTrainingObservations
        
    #     # Adiciona o array booleano ao lista
    #     trainObservations.append(trainObservations_k)
        
    #     # Indexa y[k] e W[k] usando o array booleano
    #     y_train.append(y[k][trainObservations_k])
    #     y_val.append(y[k][~trainObservations_k])
    #     W_train.append(W[k][trainObservations_k, :])
        
    #     # Transpor W_train[k] e adicionar à lista
    #     Wt_train.append(W_train[k].T)
        
    #     # Indexa W[k] usando o array booleano
    #     W_val.append(W[k][~trainObservations_k, :])
        
    #     # Prints para depuração
    #     print(k)
    #     print(f'len   {len(y)}')
    #     print(f'traink   {trainObservations_k}')
    #     print(f'train  {trainObservations}')  
    #     print(f'y_train  {y_train}')
    #     print(f'y_val  {y_val}')

    # Setup the model structure for the training subset.
    #parameterTrainingModel = model.copy()
    # parameterTrainingModel = model
    # for k in range(len(y)):
    #     observationConfidenceWeights = model.confidence[k]
    #     parameterTrainingModel.confidence[k] = observationConfidenceWeights[trainObservations[k] == 1]

    parameterTrainingModel = model
    # print(model.confidence[k])

# Update confidence weights for training subset
    for k,item in enumerate(y):
        observationConfidenceWeights = model.confidence[k]
        train_observations = trainObservations_n[k]
    # Assume parameterTrainingModel.confidence[k] is expected to be an array with the same shape as the boolean index
        # parameterTrainingModel.confidence[k] = observationConfidenceWeights[train_observations == 1]
        if observationConfidenceWeights[k]:
            parameterTrainingModel.confidence.append(item)

    # for k in range(len(y)):
    #     observationConfidenceWeights = model.confidence[k]
    #     trainObs = trainObservations[k]
    #     print(f"observationConfidenceWeights shape: {observationConfidenceWeights.shape}")
    #     print(f"trainObservations[k] type: {type(trainObs)}, shape: {trainObs.shape}")
    
    # # Check if the boolean indexing is causing the issue
    # try:
    #     filtered_weights = observationConfidenceWeights[trainObs]
    #     print(f"filtered_weights shape: {filtered_weights.shape}")
    # except Exception as e:
    #     print(f"Error during indexing: {e}")

    # # Proceed with the assignment
    # parameterTrainingModel.confidence[k] = filtered_weights

    # Define search range for adaptive grid search.
    if imagePrior.weight is not None:
        # Refine the search range from the previous iteration.
        lambdaSearchRange = np.logspace(np.log10(imagePrior.weight) - 1/maxCVIter,
                                        np.log10(imagePrior.weight) + 1/maxCVIter,
                                        maxCVIter)
    else:
        # Set search range used for initialization.
        lambdaSearchRange = np.logspace(hyperparameterCVSearchRange[0], hyperparameterCVSearchRange[1], maxCVIter)
        bestLambda = np.median(lambdaSearchRange)
    
    # Perform adaptive grid search over the selected search range.
    for lambda_ in lambdaSearchRange:
        # Estimate super-resolved image from the training set.
        parameterTrainingModel.weight = lambda_
        SR_train, numFunEvals = updateHighResolutionImage(SR, parameterTrainingModel, y_train, W_train, Wt_train,optimParams,imsize= (SR.shape[0],SR.shape[1]))
        #SR_train, numFunEvals = updateHighResolutionImage(SR,optimParams, parameterTrainingModel, y_train, W_train, Wt_train)
        
        # Determine errors on the training and the validation subset.
        valError = 0
        trainError = 0
        for k in range(len(y)):
            observationConfidenceWeights = model.confidence[k]
            # Error on the validation subset.
            valError += np.sum(observationConfidenceWeights[~trainObservations[k]] * (W_val[k] @ SR_train - y_val[k])**2)
            # Error on the training subset.
            trainError += np.sum(observationConfidenceWeights[trainObservations[k]] * (W_train[k] @ SR_train - y_train[k])**2)
        
        if valError < minValError:
            # Found optimal regularization weight.
            bestLambda = lambda_
            minValError = valError
            SR_best = SR_train
        
        # Save errors on training and validation sets if requested
        # (assuming report is a dictionary where these values are stored)
    
    
    return bestLambda, SR_best

def exist(var_name):
    try:
        eval(var_name)
        return False
    except NameError:
        return True
    

# def estimateWBTVPriorWeights(z, p=0.5, weights=None, scale_parameter=None):
#     # Estimate weights for WBTVPrior
#     if weights is None:
#         weights = np.ones_like(z)
    
#     if scale_parameter is None:
#         scale_parameter = get_adaptive_scale_parameter(z, weights)
    
#     # Estimation of weights based on pre-selected scale parameter
#     c = 2
#     scale_parameter *= c
#     weights = (p * scale_parameter**(1 - p)) / (np.abs(z)**(1 - p))
#     weights[np.abs(z) <= scale_parameter] = 1
    
#     return weights, scale_parameter

# def get_adaptive_scale_parameter(z, weights):
#     # Nested function to compute adaptive scale parameter
#     median_z = weighted_median(z, weights)
#     weighted_median_diff = np.abs(z - median_z)
#     scale_parameter = weighted_median(weighted_median_diff, weights)
#     return scale_parameter

# def weighted_median(values, weights):
#     # Function to compute weighted median
#     sorted_indices = np.argsort(values)
#     sorted_values = values[sorted_indices]
#     sorted_weights = weights[sorted_indices]
#     cumulative_weights = np.cumsum(sorted_weights)
#     total_weight = np.sum(sorted_weights)
    
#     median_index = np.searchsorted(cumulative_weights, total_weight / 2.0)
#     return sorted_values[median_index]

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

def estimateWBTVPriorWeights(z, p=0.5, weights=None, scale_parameter=None):
    # Convert input to sparse matrix if not already
    z = csr_matrix(z) if not isinstance(z, csr_matrix) else z

    if weights is None:
        weights = csr_matrix(np.ones_like(z.toarray()))  # Ensure weights is a sparse matrix
    
    if scale_parameter is None:
        scale_parameter = get_adaptive_scale_parameter(z, weights)
    
    # Estimation of weights based on pre-selected scale parameter
    c = 2
    scale_parameter *= c
    
    # Convert scale_parameter to a sparse matrix
    scale_parameter_sparse = csr_matrix(scale_parameter)
    
    # Compute weights
    abs_z = np.abs(z.toarray())
    weights = (p * scale_parameter**(1 - p)) / (abs_z**(1 - p))
    
    # Apply condition for weights based on scale_parameter
    condition = abs_z <= scale_parameter_sparse.toarray()
    weights[condition] = 1
    
    return weights, scale_parameter

def get_adaptive_scale_parameter(z, weights):
    # Convert to sparse matrices
    z = csr_matrix(z) if not isinstance(z, csr_matrix) else z
    weights = csr_matrix(weights) if not isinstance(weights, csr_matrix) else weights

    # Compute weighted median
    median_z = weighted_median(z.toarray().flatten(), weights.toarray().flatten())
    weighted_median_diff = np.abs(z.toarray().flatten() - median_z)
    scale_parameter = weighted_median(weighted_median_diff, weights.toarray().flatten())
    return scale_parameter

def weighted_median(values, weights):
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(sorted_weights)
    
    median_index = np.searchsorted(cumulative_weights, total_weight / 2.0)
    return sorted_values[median_index]


def btv_prior_weighted(x, weights, imsize, P=1, alpha=0.7):
    """
    Calculate the prior term for an image given weights and other parameters.

    Parameters:
        x (np.ndarray): The vector to be reshaped into an image.
        weights (np.ndarray): The weights used for the calculation.
        imsize (tuple): The size of the image to reshape into.
        P (int, optional): Padding size. Default is 1.
        alpha (float, optional): Decay factor. Default is 0.7.

    Returns:
        float: The computed prior value.
    """
    
    # Reshape the vector to an image
    X = vector_to_image(x, imsize)
    
    # Pad image at the border to perform shift operations
    Xpad = np.pad(X, pad_width=P, mode='symmetric')

    # Initialize prior value
    prior = 0
    k = 0
    
    # Consider shifts in the interval [-P, +P]
    for l in range(-P, P+1):
        for m in range(-P, P+1):
            if l != 0 or m != 0:
                
                # Extract the corresponding weights
                w = weights[k * (X.size):(k+ 1) * X.size]
                k += 1
                print(f'weights shape {weights.shape}')
                # Shift by l and m pixels
                Xshift = Xpad[P + l:-P + l, P + m:-P + m]

                #print(f'l shape {l.shape}')
                #print(f'm shape {m.shape}')
                print(f'X.size {X.size}')
                print(f'w shape {w.shape}')
                print(f'Xshift shape {Xshift.shape}')
                print(f'x shape {X.shape}')
                lfav = loss_fun(Xshift - X)
                print(f'loss fun{lfav.shape}')

                print(f'w type {type(w)}')
                print(f'loss fun type{type(lfav)}')

                # Compute the prior value
                prior += alpha ** (abs(l) + abs(m)) * np.sum(np.matmul(w,loss_fun(Xshift - X)))
                # prior += alpha ** (abs(l) + abs(m)) * np.sum(w * loss_fun(Xshift - X))
    
    return prior

def vector_to_image(x, imsize):
    """
    Convert a vector to an image with the given size.

    Parameters:
        x (np.ndarray): The vector to reshape.
        imsize (tuple): The size of the image to reshape into.

    Returns:
        np.ndarray: The reshaped image.
    """
    return x.reshape(imsize)

def loss_fun(x):
    """
    Calculate the loss function.

    Parameters:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The result of the loss function.
    """
    mu = 1e-4
    return np.sqrt(x**2 + mu)


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

parameterTrainingModel = Parametro()

model = Parametro()
model.P = 2
model.alpha = 0.5
model.magFactor = 2
model.psfWidth = 0.4
model.magFactor = 8
model.motionParams = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
model.confidence = []

class imagePrior(Parametro):
    def __init__(self):
        super().__init__()  
        self.weight = None
        self.size = None

imagePrior = imagePrior()

class optimParams():
    def __init__(self):
        self.maxCVIter = None
        self.maxMMiter = None
        self.sparsityParameter = None
        self.fractionCVTrainingObservations = None
        self.hyperparameterCVSearchRange = None
        self.terminationTol= None
        self.maxSCGIter = None

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
    sparsityParameter = optimParams.sparsityParameter = 0.5
    optimParams.fractionCVTrainingObservations = 0.95
    optimParams.hyperparameterCVSearchRange =[12,0]
    optimParams.terminationTol = 0.001
    optimParams.maxSCGIter = 5

    if imagePrior.weight is not None:
            useFixedRegularizationWeight = True
    else:
        useFixedRegularizationWeight = False

    for iter in range(0,optimParams.maxMMiter):
        W=[]
        Wt = []
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
                Wt = np.transpose(W)

                
                        
            if iter>0:
                
                SR= imresize.imresize(np.reshape(SR,(tensor.shape[0]*coarseToFineScaleFactors[iter-1],tensor.shape[1]*coarseToFineScaleFactors[iter-1]),'C'),
                        (coarseToFineScaleFactors[iter]/coarseToFineScaleFactors[iter-1]))
                SR = imageToVector(SR)
        
        for frameIdx in range(0,tensor.shape[2]):
            print(f'Imagem número:{frameIdx}\n')
            print(f'Y {type(Y)}')
            Y.append(imageToVector(tensor[:,:,frameIdx]))
            #residualError[frameIdx] = get_residual(SR, Y[frameIdx],W[frameIdx], model.photometricParams)
            residualError.append(get_residual(SR, Y[frameIdx],W[frameIdx], model.photometricParams)) 
            
        
        model.confidence, sigmaNoise = estimate_observation_confidence_weights(residualError, model.confidence)
        
        btvTransformedImage = getBTVTransformedImage( imresize.imresize(np.reshape(SR,(tensor.shape[0],tensor.shape[1]),'C'), model.magFactor), model.P, model.alpha);     
        if exist('btvWeights') == True:
            btvWeights= np.ones((btvTransformedImage.shape[0],btvTransformedImage.shape[1]))
        
        if len(btvWeights) != len(btvTransformedImage):
            btvWeights = imresize.imresize(btvWeights,(btvTransformedImage.shape[0],btvTransformedImage.shape[1]))

        btvWeights, sigmaBTV= estimateWBTVPriorWeights(btvTransformedImage)
        imagePrior.weight=btvWeights
        imagePrior.size= (model.magFactor*tensor.shape[0],model.magFactor*tensor.shape[1])


        #fractionCVTrainingObservations

        print(f'tipo yyyyyyyyyY {type(Y)}')

        if maxCVIter > 1 and useFixedRegularizationWeight == False:
            imagePrior.weight, SR_best = selectRegularizationWeight(SR= SR,optimParams=optimParams,model=model,y=Y,W=W)
            maxCVIter = max([round( 0.5 * maxCVIter ), 1])
        else:
            SR_best = SR

        SR_old = SR
        SR, numFunEvals = updateHighResolutionImage(SR_best, model, y, W, Wt,imsize= (SR.shape[0],SR.shape[1]))    

        if isConverged(SR, SR_old) and iter > len(coarseToFineScaleFactors):
            SR= imresize.imresize(np.reshape(SR,(tensor.shape[0],tensor.shape[1]),'C'), model.magFactor)
    
    SR = imresize.imresize(np.reshape(SR,(tensor.shape[0],tensor.shape[1]),'C'), model.magFactor)

        
