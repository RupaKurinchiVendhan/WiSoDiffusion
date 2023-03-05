import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.transform import resize
import random
import matplotlib.image as mpimg
plt.rcParams.update({'font.size': 22})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='/shared/rkurinch/results2/**')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.npy'.format(args.path), recursive=True))
    fake_names = list(glob.glob('{}/*_inf.npy'.format(args.path), recursive=True))
    bic_ua_names = list(glob.glob('{}/*_ua_bic.png'.format(args.path), recursive=True))
    bic_va_names = list(glob.glob('{}/*_va_bic.png'.format(args.path), recursive=True))

    real_names.sort()
    fake_names.sort()
    bic_ua_names.sort()
    bic_va_names.sort()

    idxs = random.sample(range(len(real_names)), 10)
    i = 0
    for idx in idxs:
        i += 1
        rname = real_names[idx]
        fname = fake_names[idx]
        bname_ua = bic_ua_names[idx]
        bname_va = bic_va_names[idx]

        rdata = np.load(rname)
        fdata = np.load(fname)

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
        axes[0,0].imshow(rdata[:,:,0])
        axes[0,0].set_ylabel('ua')
        axes[0,0].set_xlabel('Ground Truth')
        axes[0,0].xaxis.set_label_position('top')

        im = axes[0,1].imshow(fdata[:,:,0])
        axes[0,1].set_xlabel('SR3')
        axes[0,1].xaxis.set_label_position('top')

        img = Image.open(bname_ua).resize((fdata[:,:,0].shape[0], fdata[:,:,0].shape[0]), Image.BICUBIC)
        axes[0,2].imshow(img)
        axes[0,2].set_xlabel('Bicubic')
        axes[0,2].xaxis.set_label_position('top')

        axes[1,0].imshow(rdata[:,:,1])
        axes[1,0].set_ylabel('va')
        axes[1,1].imshow(fdata[:,:,1])
        img = Image.open(bname_ua).resize((fdata[:,:,0].shape[0], fdata[:,:,0].shape[0]), Image.BICUBIC)
        im = axes[1,2].imshow(img)

        for row in range(2):
            for col in range(3):
                ax = axes[row,col]
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.savefig("fig_{}.png".format(i))




