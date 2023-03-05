import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.transform import resize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='/shared/rkurinch/results2/**')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.npy'.format(args.path), recursive=True))
    fake_names = list(glob.glob('{}/*_inf.npy'.format(args.path), recursive=True))
    lr_names = list(glob.glob('{}/*_lr.npy'.format(args.path), recursive=True))

    real_names.sort()
    fake_names.sort()
    lr_names.sort()

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    for rname, fname, lname in zip(real_names, fake_names, lr_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = rname.rsplit("_inf")[0]
        lidx = rname.rsplit("_lr")[0]

        rdata = np.load(rname)
        fdata = np.load(fname)
        ldata = np.load(lname)
        ldata_ua = resize(ldata[:,:,0],(ldata.shape[0]//4,ldata.shape[1]//4))
        ldata_va = resize(ldata[:,:,1],(ldata.shape[0]//4,ldata.shape[1]//4))
        ldata = np.dstack((ldata_ua, ldata_va))
        np.save(lname.replace("_lr.npy", "_lr2.npy"), ldata)
        # bdata_ua = resize(ldata_ua, (fdata.shape[0],fdata.shape[0]))
        # bdata_va = resize(ldata_va, (fdata.shape[0],fdata.shape[0]))
        # bdata = np.dstack((bdata_ua, bdata_va))
        # np.save(lname.replace("_lr.npy", "_bic.npy"), bdata)

        minua = min(np.min(rdata[:,:,0]), np.min(fdata[:,:,0]))
        maxua = max(np.max(rdata[:,:,0]), np.max(fdata[:,:,0]))
        minva = min(np.min(rdata[:,:,1]), np.min(fdata[:,:,1]))
        maxva = max(np.max(rdata[:,:,1]), np.max(fdata[:,:,1]))

        plt.imsave(rname.replace("_hr.npy", "_ua_hr.png"), rdata[:,:,0], vmin = minua, vmax = maxua)
        plt.imsave(rname.replace("_hr.npy", "_va_hr.png"), rdata[:,:,1], vmin = minva, vmax = maxva)
        plt.imsave(fname.replace("_inf.npy", "_ua_inf.png"), fdata[:,:,0], vmin = minua, vmax = maxua)
        plt.imsave(fname.replace("_inf.npy", "_va_inf.png"), fdata[:,:,1], vmin = minva, vmax = maxva)
        # plt.imsave(lname.replace("_lr.npy", "_ua_lr.png"), ldata_ua, vmin = minua, vmax = maxua)
        # plt.imsave(lname.replace("_lr.npy", "_va_lr.png"), ldata_va, vmin = minva, vmax = maxva)
        # plt.imsave(lname.replace("_lr.npy", "_ua_bic.png"), bdata_ua, vmin = minua, vmax = maxua)
        # plt.imsave(lname.replace("_lr.npy", "_va_bic.png"), bdata_va, vmin = minva, vmax = maxva)
        lr_ua = Image.open(lname.replace("_lr.npy", "_ua_inf.png"))
        lr_va = Image.open(lname.replace("_lr.npy", "_va_inf.png"))
        lr_ua2 = lr_ua.resize((ldata.shape[0]//4, ldata.shape[1]//4), Image.BILINEAR)
        lr_va2 = lr_va.resize((ldata.shape[0]//4, ldata.shape[1]//4), Image.BILINEAR)
        lr_ua2.save(lname.replace("_lr.npy", "_ua_lr.png"))
        lr_va2.save(lname.replace("_lr.npy", "_va_lr.png"))
        bic_ua = lr_ua2.resize((ldata.shape[0], ldata.shape[1]), Image.BICUBIC)
        bic_va = lr_va2.resize((ldata.shape[0], ldata.shape[1]), Image.BICUBIC)
        bic_ua.save(lname.replace("_lr.npy", "_ua_bic.png"))
        bic_va.save(lname.replace("_lr.npy", "_va_bic.png"))




