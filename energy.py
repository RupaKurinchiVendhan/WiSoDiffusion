'''
The following code offers a method for generating a kinetic energy spectrum, in a manner similar to generating a power spectrum.
For an official implementation of how to create a plot using turbulent flow statistics as in the paper, refer to this repository:
https://github.com/b-fg/Energy_spectra/blob/master/ek.py.

Modify file directories and other parameters as necessary.
'''

import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats
import os
import glob
import argparse

Energy_Spectrum = {'Ground Truth':  {'x':[], 'y':[]}, 'Inference': {'x':[], 'y':[]}, 'Bicubic': {'x':[], 'y':[]}, 'LR': {'x':[], 'y':[]}}
COMPONENTS = {'wind': {'ua':1, 'va':1}, 'solar': {'dni':0, 'dhi':1}}

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=16) # controls default text size
plt.rc('axes', titlesize=14) # fontsize of the title
plt.rc('axes', labelsize=14) # fontsize of the x and y labels
plt.rc('xtick', labelsize=14) # fontsize of the x tick labels
plt.rc('ytick', labelsize=14) # fontsize of the y tick labels
plt.rc('legend', fontsize=14) # fontsize of the legend


def energy_spectrum(img_path):
    img = Image.open(img_path).convert('L')
    img.save('greyscale.png')
    image = mpimg.imread("greyscale.png")

    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    fourier_amplitudes = np.fft.fftshift(fourier_amplitudes)

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, Abins

def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def normalize(y, y_gt):
    for i in range(y.size):
        y[i] = y[i] + (y_gt[i] - y[i])
    return y

def rescale(y):
    min = np.min(y)
    for i in range(y.size):
        y[i] = min - (y[i] - min)
    return y

def compare_output_helper():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='/shared/rkurinch/results2/**')
    args = parser.parse_args()
    hr_names = list(glob.glob('{}/*_hr.png'.format(args.path), recursive=True))
    inf_names = list(glob.glob('{}/*_inf.png'.format(args.path), recursive=True))
    lr_names = list(glob.glob('{}/*_lr.png'.format(args.path), recursive=True))
    bic_names = list(glob.glob('{}/*_bic.png'.format(args.path), recursive=True))

    for hr_name, inf_name, lr_name, bic_name in zip(hr_names, inf_names, lr_names, bic_names):
        # gt_HR = np.load(hr_name)[:,:,0]
        # gt_INF = np.load(inf_name)[:,:,0]

        HR_kvals2, HR_ek = energy_spectrum(hr_name)
        Energy_Spectrum['Ground Truth']['x'].append(HR_kvals2)
        Energy_Spectrum['Ground Truth']['y'].append(HR_ek)

        Inf_kvals2, Inf_ek = energy_spectrum(inf_name)
        Energy_Spectrum['Inference']['x'].append(Inf_kvals2)
        Energy_Spectrum['Inference']['y'].append(Inf_ek)

        LR_kvals2, LR_ek = energy_spectrum(lr_name)
        Energy_Spectrum['LR']['x'].append(LR_kvals2)
        Energy_Spectrum['LR']['y'].append(LR_ek)
        
        Bic_kvals2, Bic_ek = energy_spectrum(bic_name)
        Energy_Spectrum['Bicubic']['x'].append(Bic_kvals2)
        Energy_Spectrum['Bicubic']['y'].append(Bic_ek)

def plot_energy_spectra():
    colors = {'Ground Truth': 'black', 'Inference': 'pink', 'LR': 'green', 'Bicubic': 'blue'}
    E_gt = np.mean(Energy_Spectrum['Ground Truth']['y'], axis=0) / 10000
    for model in Energy_Spectrum:
        k = np.flip(np.mean(Energy_Spectrum[model]['x'], axis=0))
        E = np.mean(Energy_Spectrum[model]['y'], axis=0) / 10000
        print(E[0] - E_gt[0])
        plt.loglog(k, E, color=colors[model], label=model)
    plt.xlabel("k (wavenumber)")
    plt.ylabel("Kinetic Energy")
    plt.tight_layout()
    plt.title("Energy Spectrum")
    plt.legend()
    plt.savefig("wind_spectrum2.png", dpi=1000, transparent=True, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    compare_output_helper()
    plot_energy_spectra()