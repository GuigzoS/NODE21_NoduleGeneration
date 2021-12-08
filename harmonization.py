import argparse
from SinGAN.manipulate import SinGAN_generate
from SinGAN.imresize import imresize
from SinGAN.functions import post_config, dilate_mask, convert_image_np, np2torch, read_image, adjust_scales2image, load_trained_pyramid
from torch import load
import os

def loadSG(weights_path):
    #parser = get_arguments()
    opt = get_namespace()
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

def get_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', help='task to be done', default='train')
    #workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    
    #load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z',type=int,help='noise # channels',default=1)
    parser.add_argument('--nc_im',type=int,help='image # channels',default=1)
    parser.add_argument('--out',help='output folder',default='Output')
        
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=5)#3
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)#math.floor(opt.ker_size/2)
        
    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.85)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=1024)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)
    
    #from harmonization
    parser.add_argument('--input_dir', help='input image dir', default='SinGAN/weights2')
    parser.add_argument('--input_name', help='training image name', default="n0908_crop.mha")
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Harmonization')
    parser.add_argument('--ref_name', help='reference image name', default="peuimporte.mha")
    parser.add_argument('--harmonization_start_scale', help='harmonization injection scale', type=int, default=10)
    parser.add_argument('--mode', help='task to be done', default='harmonization')
    
    return parser

def get_namespace():
    return argparse.Namespace(not_cuda=0, netG="", netD="", manualSeed=None, nc_z=1, nc_im=1, out='Output', nfc=32, min_nfc=32, ker_size=5, num_layer=5, stride=1, padd_size=0, scale_factor=0.85, noise_amp=0.1, min_size=25, max_size=1024, niter=2000, gamma=0.1, lr_g=0.0005, lr_d=0.0005, beta1=0.5, Gsteps=3, Dsteps=3, lambda_grad=0.1, alpha=10, input_dir='SinGAN/weights2', input_name="n0908_crop.mha", ref_dir='Input/Harmonization', ref_name="blabla.mha", harmonization_start_scale=10, mode='harmonization')


