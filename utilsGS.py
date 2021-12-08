import numpy as np
import random as rd
from scipy.ndimage import zoom, binary_dilation, generate_binary_structure
import os
from sklearn.cluster import KMeans

def preprocessing(arr):
    """
    Perform preprocessing on healthy patient arr
    """
    arr_norm, mini, maxi = normalise(arr)
    return arr_norm, mini, maxi

def listNodules(nodule_path, mask_path):
    """
    List and sort nodule files and corresponding masks
    """
    return [os.path.join(nodule_path, n) for n in sorted(os.listdir(nodule_path))], [os.path.join(mask_path, n) for n in sorted(os.listdir(mask_path))]

def chooseNodule(width, height, meanval, nodule_list, mask_list, ind=-1):
    """
    Select a nodule and return selected nodule and mask as arrays. TO BE IMPROVED CF HIST !!!!!!!!!!!!!!!!!!
    """
    # For the moment : choose randomly a nodule
    if ind==-1:
        ind = rd.randint(0,len(nodule_list)-1)
    else:
        print(nodule_list[ind])
    return np.load(nodule_list[ind]), np.load(mask_list[ind])

def zoomNodule(nodule, mask, width, height):
    """
    Adapt nodule size to bounding box width and height using zoom
    """
    # ADD DATA AUG TECHNIQUES HERE
    if width==0:
        width=10
    if height==0:
        height=10
    current_size = nodule.shape
    target_size = np.array([width, height])
    factor = target_size / current_size
    #print("---")
    #print(target_size)
    #print(current_size)
    #print(factor)
    nodule2 = zoom(nodule, factor, order=3)
    mask2 = zoom(mask, factor, order=0)
    mask2[mask2>0.5] = 1 #just in case
    mask2[mask2<=0.5] = 0 #just in case
    return nodule2, mask2


def pasteNodule(arr, nodule_zoom, nodulemask_zoom, y_min, x_min, y_max, x_max, bg_coef_min=0.6, bg_coef_max=0.8, nod_coef_min=0.5, nod_coef_max=0.6):
    """
    Paste nodule on healthy array and keep some background according to random coefficients
    """
    arr2 = np.copy(arr)
    nodule_zoom1024 = np.zeros(arr2.shape)
    nodule_zoom1024[x_min:x_max, y_min:y_max] = nodule_zoom
    new_mask = np.zeros(arr2.shape)
    new_mask[x_min:x_max, y_min:y_max] = nodulemask_zoom
    bg_coef = bg_coef_min + rd.random()*(bg_coef_max - bg_coef_min)
    nod_coef = nod_coef_min + rd.random()*(nod_coef_max - nod_coef_min)
    np.putmask(arr2, new_mask>0.5, bg_coef*arr + nod_coef*nodule_zoom1024)
    #arr2[x_min:x_max, y_min:y_max] = nodule_zoom
    
    return arr2, new_mask

def kmeansNodule(nodule, mask):
    """
    Perform KMeans on nodule to adjust mask
    """
    flat = nodule.flatten()[:,np.newaxis]
    kmeans = KMeans(n_clusters=2).fit(flat)
    k=0
    while kmeans.labels_[0] != 0 or k<20:
        kmeans = KMeans(n_clusters=2).fit(flat)
        k+=1
    if k==20:
        print("problem concerning KMeans")
        return mask
    else:
        seg_kmeans = kmeans.labels_.reshape(nodule.shape)
        patch_kmeans = np.zeros(nodule.shape)
        np.putmask(patch_kmeans, seg_kmeans>0, nodule)
        seg_kmeans[seg_kmeans>0] = 1
        return patch_kmeans, seg_kmeans

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

def postprocessing(arr_before_crop, arr_cropped, mini, maxi, xcrop_min, xcrop_max, ycrop_min, ycrop_max):
    """
    Perform postprocessing : decrop, denormalize and tranform harmonized image
    """
    result = np.copy(arr_before_crop)
    result[xcrop_min:xcrop_max, ycrop_min:ycrop_max] = arr_cropped
    
    return denormalise(result, mini, maxi)

def selectScale(minscale=4, maxscale=6):
    """
    Select random scale to use in SinGAN
    """
    return rd.randint(minscale, maxscale)

def transfo(arr):
    """
    Transform array to allow visualization with matplotlib. You can note that arr = transfo(transfo(arr))
    """
    return np.flip(np.rot90(arr,3), axis=1)

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

def denormalise(arr, mini, maxi):
    """
    Denormalize array : [-1,1] -> [ini,maxi]
    """
    return (arr+1)/2*(maxi-mini)+mini

def bb_correction(y_min, x_min, y_max, x_max):
    if y_max-y_min==0:
        if y_min>5:
            y_min = y_min - 5
        else:
            y_max = y_max - 5
    if x_max-x_min==0:
        if x_min>5:
            x_min = x_min - 5
        else:
            x_max = x_max - 5
    return y_min, x_min, y_max, x_max
