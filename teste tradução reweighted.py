import numpy as np
from skimage.transform import resize
import subprocess
import scipy.io as sio
from scipy.sparse import csr_matrix, coo_array
import json
import imresize
from scipy.signal import medfilt2d
#   if isempty(model.SR)
#         % Initialize super-resolved image by the temporal median of the 
#         % motion-compensated low-resolution frames.
#         %SR = imageToVector( imresize(medfilttemp(LRImages, model.motionParams), ...
#          %   coarseToFineScaleFactors(1)) );
#         SR = imageToVector( imresize(medfilttemp(LRImages, model.motionParams), ...
#             coarseToFineScaleFactors(1)) );
#     else
#         % Use the user-defined initial guess.
#         SR = imageToVector( imresize(model.SR, coarseToFineScaleFactors(1)) );
#     end   
#     % Initialize the confidence weights of the observation model.
#     for frameIdx = 1:size(LRImages, 3)
#         % Use uniform weights as initial guess.
#         model.confidence{frameIdx} = ones(numel(LRImages(:,:,frameIdx)), 1);
#     end
#     % Iterations for cross validation based hyperparameter selection.
#     maxCVIter = optimParams.maxCVIter;
    
#     % Decide if the regularization weight should be automatically adjusted
#     % per iteration.
#     if isempty(model.imagePrior.weight)
#         % No user-defined parameter available. Adjust the weight at each
#         % iteration.
#         useFixedRegularizationWeight = false;
#     else
#         % Use the user-defined regularization weight.
#         useFixedRegularizationWeight = true;
#     end
SR = []
confidenceweight = {}
optimParams = {}
model = {}
coarseToFineScaleFactors = range(min(1,model['magFactor']), model.magFactor)
Y= []
residualError = []
W = []

def imageToVector(img):
    return img.flatten('F')[:,np.newaxis]

def get_residual(SR, LR, W, photometric_params=None):
    # GETRESIDUAL Get residual error for low-resolution and super-resolved
    # data.
    # GETRESIDUAL computes the residual error caused by a super-resolved
    # image with the associated low-resolution frames and the model
    # parameters (system matrix and photometric parameters).

    if photometric_params is None:
        r = LR - np.dot(W, SR)
    else:
        num_frames = photometric_params['mult'].shape[2]
        num_lr_pixel = len(LR) // num_frames

        if np.ndim(photometric_params['mult']) == 2:  # Check if 'mult' is a vector
            bm = np.zeros_like(LR)
            ba = np.zeros_like(LR)
            for k in range(num_frames):
                start_idx = (k * num_lr_pixel)
                end_idx = ((k + 1) * num_lr_pixel)
                bm[start_idx:end_idx] = np.tile(photometric_params['mult'][:,:,k], (num_lr_pixel, 1)).flatten()
                ba[start_idx:end_idx] = np.tile(photometric_params['add'][:,:,k], (num_lr_pixel, 1)).flatten()
        else:
            bm = np.zeros_like(LR)
            ba = np.zeros_like(LR)
            for k in range(num_frames):
                start_idx = (k * num_lr_pixel)
                end_idx = ((k + 1) * num_lr_pixel)
                bm[start_idx:end_idx] = photometric_params['mult'][:,:,k].flatten()
                ba[start_idx:end_idx] = photometric_params['add'][:,:,k].flatten()

        r = LR - bm * np.dot(W, SR) - ba

    return r

def estimate_observation_confidence_weights(r, weights=None, scale_parameter=None):
    # Estimate observation confidence weights
    weights_vec = []
    r_vec = []
    r_max = 0.02  # Threshold for median residual to detect outliers
    # r_max = 0.1  # Alternative threshold (commented out in MATLAB)

    for k in range(len(r)):
        r_vec.extend(r[k])
        if np.abs(np.median(r[k])) < r_max:
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

def estimate_wbtv_prior_weights(z, p=0.5, weights=None, scale_parameter=None):
    # Estimate weights for WBTVPrior
    if weights is None:
        weights = np.ones_like(z)
    
    if scale_parameter is None:
        scale_parameter = get_adaptive_scale_parameter(z, weights)
    
    # Estimation of weights based on pre-selected scale parameter
    c = 2
    scale_parameter *= c
    weights = (p * scale_parameter**(1 - p)) / (np.abs(z)**(1 - p))
    weights[np.abs(z) <= scale_parameter] = 1
    
    return weights, scale_parameter

def get_adaptive_scale_parameter(z, weights):
    # Nested function to compute adaptive scale parameter
    median_z = weighted_median(z, weights)
    weighted_median_diff = np.abs(z - median_z)
    scale_parameter = weighted_median(weighted_median_diff, weights)
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

SR = imageToVector(SR)

for frameIdx in range(0,tensor.shape[2]):
    confidenceweight[frameIdx] = np.ones(len(tensor.shape[:,:,frameIdx],1))

maxCVIter = optimParams['maxCVIter']

if model['imagePrior'] and model['imagePrior'].get('weight', None) is not None:
        useFixedRegularizationWeight = True
else:
    useFixedRegularizationWeight = False

for iter in range(0,optimParams['maxMMiter']):
    if iter <= len(coarseToFineScaleFactors):
        model['magFactor'] = coarseToFineScaleFactors[iter]

        for frameIdx in range(0,tensor.shape[2]):
            params = {'size': tensor.shape[:,:,frameIdx], 'magFactor': model['magFactor'], 'psfWidth' :  model['psfWidth'], 'motionParams':  model['motionParams'] }
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
            W[frameIdx] = sio.loadmat('Wmx.mat')

    elif iter>1:
        SR= imresize.imresize(np.reshape(SR,(tensor.shape[0]*coarseToFineScaleFactors[iter-1],tensor.shape[1]*coarseToFineScaleFactors[iter-1]),'C'),
                     (tensor.shape[0]*coarseToFineScaleFactors[iter],tensor.shape[1]*coarseToFineScaleFactors[iter]))
        SR = imageToVector(SR)
         

    for frameIdx in range(0,tensor.shape[2]):
        Y[frameIdx] = imageToVector(tensor[:,:,frameIdx])
        residualError[frameIdx] = get_residual(SR, Y[frameIdx],W[frameIdx], model['photometric_params'])
  
    model['confidence'], sigmanoise = estimate_observation_confidence_weights(residualError, model['confidence'])
    ################################
    ################################
    ################################
    #model.imagePrior.parameters{3}, model.imagePrior.parameters{4}
    btvTransformedImage = get_btv_transformed_image(np.reshape(SR,(model['magFactor']*tensor[:,:,0].shape[0],model['magFactor']*tensor[:,:,0].shape[0],'C')))

    try:
        btvWeights
    except NameError:
        btvWeights = np.ones(btvTransformedImage.shape[0],btvTransformedImage.shape[1])

    if (btvWeights.shape[0]*btvWeights.shape[1]) != (btvTransformedImage.shape[0]*btvTransformedImage.shape[1]):
        btvWeights = imresize.imresize(btvWeights,(btvTransformedImage.shape[0]*btvTransformedImage.shape[1]))

    btvWeights, sigmaBTV = estimate_wbtv_prior_weights(btvTransformedImage,optimParams['sparsityParameter'],btvWeights)

    if maxCVIter > 1 and useFixedRegularizationWeight == False:
        model.imagePrior.weight, SR_best = selectregularizationweight() 
        maxCVIter = max(round(0.5 * maxCVIter), 1)
    else:
        # No automatic hyperparameter selection required
        SR_best = SR
    
    SR_old = SR
    SR, numFumEvals =