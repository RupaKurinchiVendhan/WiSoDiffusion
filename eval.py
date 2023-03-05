import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='/shared/rkurinch/results2/**')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path), recursive=True))
    fake_names = list(glob.glob('{}/*_bic.png'.format(args.path), recursive=True))
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
        fidx = rname.rsplit("_bic")[0]
        # assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(
        #    ridx, fidx)

        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))

        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        # ssim = Metrics.calculate_ssim(sr_img, hr_img)
        mse = Metrics.calculate_mse(sr_img, hr_img)
        mae = Metrics.calculate_mae(sr_img, hr_img)

        # avg_ssim += ssim
        if psnr != float('inf'):
            avg_psnr += psnr
            # avg_ssim += ssim
            avg_mse += mse
            avg_mae += mae
    avg_psnr = avg_psnr / idx
    # avg_ssim = avg_ssim / idx
    avg_mse = avg_mse / idx
    avg_mae = avg_mae / idx

    # log
    print('# Validation # PSNR: {}'.format(avg_psnr))
    print('# Validation # SSIM: {}'.format(avg_ssim))
    print('# Validation # MSE: {}'.format(avg_mse))
    print('# Validation # MAE: {}'.format(avg_mae))
