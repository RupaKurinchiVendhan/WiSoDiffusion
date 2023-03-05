import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='/shared/rkurinch/sr3_n=100000_dr=0.2_lr=3e-06_1/**')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.npy'.format(args.path), recursive=True))
    fake_names = list(glob.glob('{}/*_sr.npy'.format(args.path), recursive=True))
    real_names.sort()
    fake_names.sort()

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_mse = 0.0
    avg_mae = 0.0
    idx = 0
    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = rname.rsplit("_sr")[0]

        hr_img = np.load(rname)
        sr_img = np.load(fname)

        mse = Metrics.calculate_mse(sr_img, hr_img)
        mae = Metrics.calculate_mae(sr_img, hr_img)

        # avg_ssim += ssim
        if mse != float('inf'):
            avg_mse += mse
            avg_mae += mae
    # avg_ssim = avg_ssim / idx
    avg_mse = avg_mse / idx
    avg_mae = avg_mae / idx

    # log
    print('# Validation # MSE: {}'.format(avg_mse))
    print('# Validation # MAE: {}'.format(avg_mae))
