import argparse
from SinGAN.manipulate import SinGAN_generate
from SinGAN.imresize import imresize
from SinGAN.functions import post_config, dilate_mask, convert_image_np, np2torch, read_image, adjust_scales2image, load_trained_pyramid
from torch import load
import os

def loadSG(weights_path, case):
    if case==1:
        opt = get_namespace1()
    else:
        opt = get_namespace2()
    opt = post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    real = read_image(opt)
    real = adjust_scales2image(real, opt)
    Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt) #weights_path[0])
    return opt, Gs, Zs, reals, NoiseAmp, real


def infer(opt, Gs, Zs, reals, NoiseAmp, real, scale, ref, mask): #n=scale
    N = len(reals) - 1
    mini = ref.min()
    maxi = ref.max()
    ref = np2torch(ref,opt)
    ref = ref[:,0:3,:,:]
    mask = np2torch(mask,opt)
    mask = mask[:,0:3,:,:]
    mask = dilate_mask(mask, opt)
    in_s = imresize(ref, pow(opt.scale_factor, (N - scale + 1)), opt)
    in_s = in_s[:, :, :reals[scale - 1].shape[2], :reals[scale - 1].shape[3]]
    in_s = imresize(in_s, 1 / opt.scale_factor, opt)
    in_s = in_s[:, :, :reals[scale].shape[2], :reals[scale].shape[3]]
    out = SinGAN_generate(Gs[scale:], Zs[scale:], reals, NoiseAmp[scale:], opt, in_s, n=scale, num_samples=1)
    out = (1-mask)*ref+mask*out
    harmonized = convert_image_np(out.detach(), mini, maxi)
    #print("-"*10)
    #print(opt.case)
    #print(harmonized.min())
    #print(harmonized.max())
    #if harmonized.min() != mini or harmonized.max() != maxi:
    #    harmonized = (harmonized-harmonized.min())/(harmonized.max()-harmonized.min())*(maxi-mini)+mini
    return harmonized
"""
def load_trained_pyramid(path):
    if(os.path.exists(path)):
        Gs = load('%s/Gs.pth' % path)
        Zs = load('%s/Zs.pth' % path)
        reals = load('%s/reals.pth' % path)
        NoiseAmp = load('%s/NoiseAmp.pth' % path)
    else:
        print('no appropriate trained model is exist, please train first')
    return Gs,Zs,reals,NoiseAmp"""


def get_namespace1():
    return argparse.Namespace(not_cuda=0, netG="", netD="", manualSeed=None, nc_z=1, nc_im=1, out='Output', nfc=32, min_nfc=32, ker_size=5, num_layer=5, stride=1, padd_size=0, scale_factor=0.85, noise_amp=0.1, min_size=25, max_size=1024, niter=2000, gamma=0.1, lr_g=0.0005, lr_d=0.0005, beta1=0.5, Gsteps=3, Dsteps=3, lambda_grad=0.1, alpha=10, input_dir='SinGAN/weights1', input_name="n0908_crop.mha", ref_dir='Input/Harmonization', ref_name="blabla.mha", harmonization_start_scale=10, mode='harmonization', case=1)


def get_namespace2():
    return argparse.Namespace(not_cuda=0, netG="", netD="", manualSeed=None, nc_z=1, nc_im=1, out='Output', nfc=32, min_nfc=32, ker_size=5, num_layer=5, stride=1, padd_size=0, scale_factor=0.85, noise_amp=0.1, min_size=25, max_size=1024, niter=2000, gamma=0.1, lr_g=0.0005, lr_d=0.0005, beta1=0.5, Gsteps=3, Dsteps=3, lambda_grad=0.1, alpha=10, input_dir='SinGAN/weights2', input_name="n0301_crop.mha", ref_dir='Input/Harmonization', ref_name="blabla.mha", harmonization_start_scale=10, mode='harmonization', case=2)


