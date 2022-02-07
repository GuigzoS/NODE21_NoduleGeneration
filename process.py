import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import utilsGS as guigz # Main author : Guillaume SALLE aka Guigz (also baseline's code)
import harmonization as sg # SinGAN codes
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw
import pandas as pd
import os
import random

# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = True
class Nodulegeneration(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path("/input/") if execute_in_docker else Path("./test/"),
            output_path = Path("/output/") if execute_in_docker else Path("./results_loc/"),
            output_file = Path("/output/results.json") if execute_in_docker else Path("./results_loc/results.json")
        )

        # load nodules.json for location
        with open("/input/nodules.json" if execute_in_docker else "./test/nodules.json") as f:
            self.data = json.load(f)
        
        # load SinGAN weights
        sg_weights1 = Path("/opt/algorithm/SinGAN/weights1/") if execute_in_docker else Path("./SinGAN/weights1/")
        self.opt1, self.Gs1, self.Zs1, self.reals1, self.NoiseAmp1, self.real1 = sg.loadSG(sg_weights1, 1)
        sg_weights2 = Path("/opt/algorithm/SinGAN/weights2/") if execute_in_docker else Path("./SinGAN/weights2/")
        self.opt2, self.Gs2, self.Zs2, self.reals2, self.NoiseAmp2, self.real2 = sg.loadSG(sg_weights2, 2)
        
        # load info about CT_patch
        self.path_nodule = '/opt/algorithm/patch_v7nokmeans/' if execute_in_docker else './patch_v7nokmeans'
        self.pd_data = pd.read_csv('/opt/algorithm/ct_nodules_v7nokmeans.csv' if execute_in_docker else "./ct_nodules_v7nokmeans.csv")
        
    
    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        input_image = SimpleITK.GetArrayFromImage(input_image)
        total_time = time.time()
        if len(input_image.shape)==2:
            input_image = np.expand_dims(input_image, 0)
        
        nodule_images = np.zeros(input_image.shape)
        for j in range(len(input_image)):
            t = time.time()
            arr = np.copy(input_image[j,:,:])
            nodule_data = [i for i in self.data['boxes'] if i['corners'][0][2]==j]

            for nodule in nodule_data:
                # Normalize and transform 
                arr_ori = np.copy(arr)
                arr, mini, maxi = guigz.preprocessing(arr)
                # Get bounding box coordinates (correct it if it goes out of the array or if the size is 0)
                boxes = nodule['corners']
                y_min, x_min, y_max, x_max = int(boxes[2][0]), int(boxes[2][1]), int(boxes[0][0]), int(boxes[0][1])
                y_min, x_min, y_max, x_max = guigz.bb_correction(y_min, x_min, y_max, x_max)
                # Choose and load CT nodule with corresponding mask randomly with close diameter (FROM BASELINE)
                required_diameter = min(x_max-x_min, y_max-y_min)
                nodule, nodulemask, diameter, nodule_name = guigz.chooseNodule(self.pd_data, self.path_nodule, required_diameter)
                # Zoom nodule to match bounding box dimension (include data augmentation such as random flip)
                nodule_zoom, nodulemask_zoom = guigz.zoomNodule(nodule, nodulemask, (x_min, x_max), (y_min, y_max), x_max-x_min, y_max-y_min)
                # Use Otsu Threshold to choose SinGAN weights : hyposignal location -> weights1 ; hypersignal location -> weights2
                case, threshold = guigz.otsuDecidesCase(arr, y_min, x_min, y_max, x_max)
                # Contrast matching : set nodule between 0 and 1, match contrast, and put again between -1 and 1
                nodule_contrasted = guigz.matchContrast(nodule_zoom, (arr[x_min:x_max, y_min:y_max]+1)/2, case)
                # Paste nodule on CXR
                arr_with_nodule, mask_with_nodule = guigz.pasteNodule(arr, nodule_contrasted, nodulemask_zoom, y_min, x_min, y_max, x_max)
                # Create 256 crop and save this crop without the pasted nodule for later
                arr_crop, mask_crop, xcrop_min, xcrop_max, ycrop_min, ycrop_max = guigz.cropCXR(arr_with_nodule, mask_with_nodule, y_min, x_min, y_max, x_max)
                arr_crop_nonodule = arr[xcrop_min:xcrop_max, ycrop_min:ycrop_max]
                # Dilate mask
                mask_dilated = guigz.dilateNoduleMask(mask_crop)
                # Choose harmonization scale for SinGAN and infer in the given case
                scale = guigz.selectScale(case)
                # Infer in the given case
                if case==1:
                    SGresult = sg.infer(self.opt1, self.Gs1, self.Zs1, self.reals1, self.NoiseAmp1, self.real1, scale, arr_crop, mask_dilated)
                else:
                    SGresult = sg.infer(self.opt2, self.Gs2, self.Zs2, self.reals2, self.NoiseAmp2, self.real2, scale, arr_crop, mask_dilated)
                # Apply weighted mean between background and nodule
                coef_harmo, coef_arr = guigz.coefWeightedMean(case, x_max-x_min, y_max-y_min)
                SGresult = ((arr_crop_nonodule+maxi)*coef_arr + (SGresult+maxi)*coef_harmo)/(coef_arr+coef_harmo)-maxi # TEST IF I CAN REMOVE MAXI
                # Decrop and put into original format 
                arr = guigz.postprocessing(arr_with_nodule, SGresult, mini, maxi, xcrop_min, xcrop_max, ycrop_min, ycrop_max)
                
            # Put output between 0 and 255 as the baseline do (it also creates smaller mha file)
            final_result = arr/arr.max()*255
            # Add some Gaussian noise, blurry using gaussian kernel, and add a second time Gaussian noise
            # Otherwise, the detection network overfit the data very very fast
            final_result = guigz.addNoiseAndBlurring(final_result)
            # Put the final result in the 3D volume
            nodule_images[j,:,:] = final_result
            
        t_end = time.time()
        print('total time taken ', t_end-total_time)
        print('time taken per image ', (t_end-total_time)/len(input_image))
        return SimpleITK.GetImageFromArray(nodule_images)

if __name__ == "__main__":
    Nodulegeneration().process()

#['1.3.6.1.4.1.14519.5.2.1.6279.6001.195557219224169985110295082004_dcm_4.npy', '1.3.6.1.4.1.14519.5.2.1.6279.6001.271220641987745483198036913951_dcm_0.npy', 13, False] dans pack de 100
#['1.3.6.1.4.1.14519.5.2.1.6279.6001.326057189095429101398977448288_dcm_1.npy', '1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518_dcm_11.npy', 105, False] dans pack de 400_1
#['1.3.6.1.4.1.14519.5.2.1.6279.6001.202283133206014258077705539227_dcm_3.npy', '1.3.6.1.4.1.14519.5.2.1.6279.6001.117040183261056772902616195387_dcm_1.npy', 163, False] dans pack de 400_1
#['1.3.6.1.4.1.14519.5.2.1.6279.6001.392861216720727557882279374324_dcm_0.npy', '1.3.6.1.4.1.14519.5.2.1.6279.6001.250863365157630276148828903732_dcm_0.npy', 183, False] dans pack 400_1

# OLD
"""
            if TEST:
                im = Image.fromarray((guigz.normalise(arr)[0]+1)/2*255).convert("L")
                draw = ImageDraw.Draw(im)
                for nodule in nodule_data:
                    boxes = nodule['corners']
                    y_min, x_min, y_max, x_max = int(boxes[2][0]), int(boxes[2][1]), int(boxes[0][0]), int(boxes[0][1])
                    draw.rectangle([y_min, x_min, y_max, x_max],width=2)
                im.save(output_for_png+"/"+format(j,"04d")+".png")
            
            # Put output between 0 and 255 as the baseline do (it also creates smaller mha file)
            nodule_images[j,:,:] = np.around(arr/arr.max()*255) 
        print(nodule_images.shape)
        print(type(nodule_images[3,10,10]))
        print(nodule_images.dtype)
        t_end = time.time()
        print('total time taken ', t_end-total_time)
        print('time taken per image ', (t_end-total_time)/len(input_image))
        print('total case 1 : ', tot_case1)
        print('total case 2 : ', tot_case2)
        return SimpleITK.GetImageFromArray(nodule_images)

"""
