import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import utilsGS as guigz
import harmonization as sg
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw
#import pandas as pd
import os

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
            input_path = Path("/input/") if execute_in_docker else Path("./test_full/"),
            output_path = Path("/output/") if execute_in_docker else Path("./output_full/"),
            output_file = Path("/output/results.json") if execute_in_docker else Path("./output/results.json")

        )

        # load nodules.json for location
        with open("/input/nodules.json" if execute_in_docker else "test_full/nodules.json") as f:
            self.data = json.load(f)
        
        # load SinGAN weights
        sg_weights = Path("/SinGAN/weights2/") if execute_in_docker else Path("./SinGAN/weights2/"),
        self.opt, self.Gs, self.Zs, self.reals, self.NoiseAmp, self.real = sg.loadSG(sg_weights)
        
        # load info about CT_patch ????
        self.ct_nodules_dir = Path("/cxr_patch/nodule_stacked") if execute_in_docker else Path("./cxr_patch/nodule_stacked")
        self.ct_nodulemasks_dir = Path("/cxr_patch/nodule_stacked_seg") if execute_in_docker else Path("./cxr_patch/nodule_stacked_seg")
        self.nodule_list, self.nodule_mask_list = guigz.listNodules(self.ct_nodules_dir, self.ct_nodulemasks_dir)
        
    
    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        TEST = False
        KMEANS = False
        
        if TEST:
            os.makedirs("./output_full", exist_ok=True)
            os.makedirs("./output_png", exist_ok=True)
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
                #print("------------")
                # Normalize and transform 
                arr, mini, maxi = guigz.preprocessing(arr)
                # Get bounding box coordinates
                boxes = nodule['corners']
                y_min, x_min, y_max, x_max = int(boxes[2][0]), int(boxes[2][1]), int(boxes[0][0]), int(boxes[0][1])
                y_min, x_min, y_max, x_max = guigz.bb_correction(y_min, x_min, y_max, x_max)
                #print(y_min, x_min, y_max, x_max)
                # Choose and load CT nodule with corresponding mask
                meanval = np.mean(arr[x_min:x_max+1, y_min:y_max+1])
                nodule, nodulemask = guigz.chooseNodule(x_max-x_min, y_max-y_min, meanval, self.nodule_list, self.nodule_mask_list)
                # Zoom nodule to match bounding box dimension (include data augmentation such as random flip)
                nodule_zoom, nodulemask_zoom = guigz.zoomNodule(nodule, nodulemask, x_max-x_min, y_max-y_min)
                # Apply KMeans to readjust mask ############# IF IT WASN'T DONE BEFORE 
                if KMEANS:
                    nodule_zoom, nodulemask_zoom = guigz.kmeansNodule(nodule_zoom, nodulemask_zoom)
                # Paste nodule on CXR
                arr_with_nodule, mask_with_nodule = guigz.pasteNodule(arr, nodule_zoom, nodulemask_zoom, y_min, x_min, y_max, x_max)
                # Create 256 crop
                arr_crop, mask_crop, xcrop_min, xcrop_max, ycrop_min, ycrop_max = guigz.cropCXR(arr_with_nodule, mask_with_nodule, y_min, x_min, y_max, x_max)
                # Dilate mask
                mask_dilated = guigz.dilateNoduleMask(mask_crop)
                # Infer SinGAN
                n = guigz.selectScale()
                SGresult = sg.infer(self.opt, self.Gs, self.Zs, self.reals, self.NoiseAmp, self.real, n, arr_crop, mask_dilated)
                # Put into original format : decrop, denormalize and detransform
                arr = guigz.postprocessing(arr_with_nodule, SGresult, mini, maxi, xcrop_min, xcrop_max, ycrop_min, ycrop_max)
            if TEST:
                im = Image.fromarray((guigz.normalise(arr)[0]+1)/2*255).convert("L")
                draw = ImageDraw.Draw(im)
                for nodule in nodule_data:
                    boxes = nodule['corners']
                    y_min, x_min, y_max, x_max = int(boxes[2][0]), int(boxes[2][1]), int(boxes[0][0]), int(boxes[0][1])
                    draw.rectangle([y_min, x_min, y_max, x_max],width=2)
                im.save("output_png/"+format(j,"04d")+".png")
            nodule_images[j,:,:] = arr
        t_end = time.time()
        print('total time took ', t_end-total_time)
        print('time taken per image ', (t_end-total_time)/len(input_image))
        return SimpleITK.GetImageFromArray(nodule_images)

if __name__ == "__main__":
    Nodulegeneration().process()

                                                                 
                                                                 
