import numpy as np
import random as rd
from scipy.ndimage import zoom, binary_dilation, generate_binary_structure
import os
from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu
from skimage.exposure import match_histograms
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

def normalise(arr, mini=None, maxi=None):
    """
    Normalize array using window [mini,maxi] (arr.min() and arr.max() if not specified)
    """
    if mini==None:
        mini=arr.min()
    if maxi==None:
        maxi=arr.max()
    arr2 = (arr-mini)/(maxi-mini)*2-1.0
    arr2[arr2>1]=1.0
    arr2[arr2<-1]=-1.0
    return arr2, arr.min(), arr.max()

def denormalize(arr, mini, maxi):
    """
    Denormalize array : [-1,1] -> [ini,maxi]
    """
    return (arr+arr.max())/(arr.max()-arr.min())*(maxi-mini)+mini

def preprocessing(arr):
    """
    Perform preprocessing on healthy patient arr
    """
    arr_norm, mini, maxi = normalise(arr)
    return arr_norm, mini, maxi

def bb_correction(y_min, x_min, y_max, x_max):
    """
    Correct bounding box of size 0
    """
    if x_min<0:
        x_min=0
    if y_min<0:
        y_min=0
    if x_max>1024:
        x_max=1024
    if y_max>1024:
        y_max=1024
    if y_max-y_min==0:
        if y_min>5:
            y_min = y_min - 5
        else:
            y_max = y_max + 5
    if x_max-x_min==0:
        if x_min>5:
            x_min = x_min - 5
        else:
            x_max = x_max + 5
    return y_min, x_min, y_max, x_max

def chooseNodule(pd_data, path_nodule, required_diameter):
    """
    Select a nodule and return selected nodule and mask as arrays (same as generation baseline)
    """
    # For the moment : choose randomly a nodule
    
    ct_list = pd_data[pd_data['diameter']>int((required_diameter/5))].values
    if len(ct_list)<1:
        ct_list = pd_data[pd_data['diameter']>int((required_diameter/10))].values
    if len(ct_list)<1:
        ct_list = pd_data[pd_data['diameter']>int((required_diameter/20))].values
    if len(ct_list)<1:
        ct_list = pd_data.values
    index_ct = rd.randint(0, len(ct_list)-1)
    nod_name = ct_list[index_ct][1][:-3]+"npy"
    seg_name = ct_list[index_ct][1].replace('dcm','seg')[:-3]+"npy"
    diameter = ct_list[index_ct][2]
    nodule = np.load(os.path.join(path_nodule,nod_name))
    nodulemask = np.load(os.path.join(path_nodule,seg_name))
    return nodule, nodulemask, diameter, nod_name

def zoomNodule(nodule, mask,  x, y, width, height):
    """
    Adapt nodule size to bounding box width and height using ndimage.zoom
    """
    if width==0:
        width=10
    if height==0:
        height=10
    current_size = max(nodule.shape) # = nodule.shape[1]
    target_size = min(width, height)
    factor = target_size / current_size
    nodule2 = zoom(nodule, factor, order=3)
    mask2 = zoom(mask, factor, order=0)
    nodule2[nodule2>1]=1.
    nodule2[nodule2<-1]=-1.
    mask2[mask2>0.5] = 1 #just in case
    mask2[mask2<=0.5] = 0 #just in case
    if nodule2.shape != (width, height):
        nodule3 = np.zeros((width, height))
        mask3 = np.zeros((width, height))
        nodule3[:nodule2.shape[0], :nodule2.shape[1]] = nodule2
        mask3[:nodule2.shape[0], :nodule2.shape[1]] = mask2
        #print("zoomNodule nodule3.shape = "+str(nodule3.shape)+" when shape ("+str(width)+","+str(height)+") is wanted (coordinates "+str(x)+","+str(y)+")")
        return nodule3, mask3
    #print("zoomNodule nodule2.shape = "+str(nodule2.shape)+" when shape ("+str(width)+","+str(height)+") is wanted (coordinates "+str(x)+","+str(y)+")")
    return nodule2, mask2

def otsuDecidesCase(arr, y_min, x_min, y_max, x_max, debug=[], proportion=0.5):
    """
    Compute Otsu threshold with 4 classes to decide if we use SinGAN first weights or second weights (case 1 or case 2).
    If the bounding box is mainly hypersignal, then we select second weights, otherwise first weights
    """
    arr_255 = ((arr+1)/2*255).astype(int)
    if np.min(arr_255)==np.max(arr_255):
        print("une seule valeur dans l'array")
        print(debug)
        return 1, 255
    otsu_value = threshold_multiotsu(arr_255, 4)
    
    otsu_value = otsu_value[-1]
    crop = arr_255[x_min:x_max, y_min:y_max]
    mean = np.mean((crop>otsu_value).astype(int)) #between 0 and 1 -> compare to proportion
    if mean<proportion:
        return 1, otsu_value
    else:
        return 2, otsu_value

def matchContrast(nodule_2d, lung_photo, case):
    """
     Contrast matching according to Litjens et al.
     With some additional clip to prevent negative values or 0.
      nodule_2d: intensities of the nodule
      lung_photo: intensities of this particular lung area
     returns c, but is clipped to a different value according to background mean value since low values made the nodules neigh invisible sometimes.
    """
    nodule_2d = (nodule_2d+1)/2
    # mean from only nodule pixels
    indexes = nodule_2d != np.min(nodule_2d)
    it = np.mean(nodule_2d[indexes].flatten())

    # mean of the surrounding lung tissue
    ib = np.mean(lung_photo.flatten())
    
    # determine contrast
    c = np.log(it/ib)
    if case==1:
        c = max(0.6, c)
    else:
        c = max(0.7, c)
    nodule_contrasted = nodule_2d * c #+ 20/255
    nodule_contrasted[nodule_contrasted>1]=1
    nodule_contrasted[nodule_contrasted<0]=0
    nodule_contrasted = nodule_contrasted*2-1
    return nodule_contrasted

def pasteNodule(arr, nodule_zoom, nodulemask_zoom, y_min, x_min, y_max, x_max):#, bg_coef_min=1.0, bg_coef_max=1.0, nod_coef_min=1.0, nod_coef_max=1.0):
    """
    Paste nodule on healthy array and keep some background according to random coefficients
    """
    arr2 = np.copy(arr)
    nodule_zoom1024 = np.zeros(arr2.shape)
    nodule_zoom1024[x_min:x_max, y_min:y_max] = nodule_zoom
    new_mask = np.zeros(arr2.shape)
    new_mask[x_min:x_max, y_min:y_max] = nodulemask_zoom
    #bg_coef = bg_coef_min + rd.random()*(bg_coef_max - bg_coef_min)
    #nod_coef = nod_coef_min + rd.random()*(nod_coef_max - nod_coef_min)
    
    arr = (arr+1)/2
    nodule_zoom1024 = (nodule_zoom1024+1)/2
    
    #to_paste = bg_coef*arr + nod_coef*nodule_zoom1024
    to_paste = arr + nodule_zoom1024 #+ 0.2
    if not(np.isnan(to_paste).any()):
        to_paste[to_paste>1.0]=1.0
        to_paste[to_paste<0.0]=0.0
    #to_paste = sum_hypersignal(arr, nodule_zoom1024)
        arr2 = (arr2+1)/2
        np.putmask(arr2, new_mask>0.5, to_paste)
        arr2 = arr2*2-1
    else :
        arr2 = np.copy(arr)
        new_mask = np.zeros(arr2.shape)
    return arr2, new_mask


def dilateNoduleMask(mask):
    """
    Dilate mask on healthy array for SinGAN harmonization
    """
    struct = generate_binary_structure(2, 2)
    return binary_dilation(mask, structure=struct, iterations=1).astype(mask.dtype)

def cropCXR_index(y_min, x_min, y_max, x_max):
    """
    Choose indexes for cropping a 256 square containing the entire bounding box
    """
    width = x_max - x_min
    height = y_max - y_min
    if x_min + width//2 > 128 and 1024-x_max+width//2 > 128: #on peut centrer
        xcrop_min = x_min + width//2 - 128
        xcrop_max = xcrop_min+256
    elif x_min + width//2 > 128:
        xcrop_min = 768
        xcrop_max = 1024
    elif 1024-x_max+width//2 > 128:
        xcrop_min = 0
        xcrop_max = 256
    if y_min + height//2 > 128 and 1024-y_max+height//2 > 128: #on peut centrer
        ycrop_min = y_min + height//2 - 128
        ycrop_max = ycrop_min+256
    elif y_min + height//2 > 128:
        ycrop_min = 768
        ycrop_max = 1024
    elif 1024-y_max+height//2 > 128:
        ycrop_min = 0
        ycrop_max = 256
    return xcrop_min, xcrop_max, ycrop_min, ycrop_max

def cropCXR(arr_with_nodule, mask_with_nodule, y_min, x_min, y_max, x_max):
    """
    Crop healthy arr around bounding box for SinGAN inference
    """
    xcrop_min, xcrop_max, ycrop_min, ycrop_max = cropCXR_index(y_min, x_min, y_max, x_max)
    return arr_with_nodule[xcrop_min:xcrop_max, ycrop_min:ycrop_max], mask_with_nodule[xcrop_min:xcrop_max, ycrop_min:ycrop_max], xcrop_min, xcrop_max, ycrop_min, ycrop_max

def selectScale(case):
    """
    Select random scale to use in SinGAN
    """
    if case==1:
        scale = rd.randint(5, 7)
    else:
        scale = rd.randint(7, 9)
    
    return scale

def coefWeightedMean(case, diff_x, diff_y):
    """
    Return coefficients for weighted mean after SinGAN inference.
    For the moment it is random in a fine-tuned window for each case.
    It may be better if adapted according to the nodule size.
    """
    if case==1:
        coef_harmo = rd.randint(45,55)
    else:
        coef_harmo = rd.randint(35,45)
    return coef_harmo, 100 - coef_harmo

def postprocessing(arr_before_crop, arr_cropped, mini, maxi, xcrop_min, xcrop_max, ycrop_min, ycrop_max):
    """
    Perform postprocessing : decrop, denormalize and tranform harmonized image
    """
    result = np.copy(arr_before_crop)
    result[xcrop_min:xcrop_max, ycrop_min:ycrop_max] = arr_cropped
    
    return denormalize(result, mini, maxi)

def addNoiseAndBlurring(arr, stdGaussianFilter_min=0.9, stdGaussianFilter_max=1.1, stdGaussianNoise_min=2.8, stdGaussianNoise_max=3.2, kernel_mode="automatic"):
    """
    Add some Gaussian noise, blurry using gaussian kernel, and add a second time Gaussian noise
    It avoids overfitting in the detection model. It can also be seen as a specific kind of data augmentation.
    """
    arr = arr + np.random.normal(0, rd.random()*(stdGaussianNoise_max-stdGaussianNoise_min)+stdGaussianNoise_min, arr.shape)
    
    blurry_done = False
    if kernel_mode=="automatic":
        sigma = rd.random()*(stdGaussianFilter_max-stdGaussianFilter_min)+stdGaussianFilter_min
        arr = gaussian_filter(arr, sigma=sigma)
        blurry_done = True
    elif kernel_mode=="gaussian_blur_5":
        kernel = 1/256*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
    elif kernel_mode=="gaussian_blur_3":
        kernel = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])
    elif kernel_mode=="mean_blur_3":
        kernel = 1/9*np.array([[1,1,1],[1,1,1],[1,1,1]])
    elif kernel_mode=="mean_blur_3":
        kernel = 1/9*np.array([[1,1,1],[1,1,1],[1,1,1]])
    else:
        print("addNoiseAndBlurring : don't know which kernel to use -> use gaussian_blur_5")
        kernel = 1/256*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
    
    if not(blurry_done):
        arr = convolve2d(arr, kernel, boundary='symm', mode='same')
    arr = arr + np.random.normal(0, rd.random()*(stdGaussianNoise_max-stdGaussianNoise_min)+stdGaussianNoise_min, arr.shape)
    arr[arr<0]=0
    arr[arr>255]=255
    return np.around(arr)




