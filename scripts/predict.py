from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import os

import pywt
import scipy
from tqdm.auto import tqdm
import numpy as np
import nibabel as nb
from scipy.ndimage import zoom

cores = int(cpu_count() / 2)  # be carefull with RAM usage in abdom


def load_and_prep_nifti(fname, path_inp, data: str):
    nii = nb.load(path_inp / fname)

    img = nii.get_fdata()
    spacing = np.asarray(nii.header['pixdim'][1:4], dtype='float32')

    img = min_max_scale_q(img)

    new_spacing = (0.7, 0.7875, 0.7) if (data == 'brain') else (0.75, 0.75, 0.8351563)  # if data == 'abdominal'
    img = zoom_image(img, spacing, new_spacing=new_spacing)
    img = min_max_scale(img)
    return img


def compute_histogram(img, n_bins=150):
    histogram, _ = np.histogram(img, bins=n_bins, range=(0, 1), density=True)
    texture_histogram = compute_wavelet_features(img)
    return np.concatenate((histogram, texture_histogram))


def get_ood_score(fname, path_inp, data: str, embeddings: np.ndarray, n_bins: int = 150):
    img = load_and_prep_nifti(fname, path_inp, data)

    # ### OOD score ###
    histogram = compute_histogram(img, n_bins=n_bins)
    mean_mahalanobis = embeddings.mean(axis=0)

    demeaned = embeddings - mean_mahalanobis
    cov_mahalanobis = demeaned.T @ demeaned / len(embeddings)

    inv_covariance_mahalanobis = np.linalg.inv(cov_mahalanobis)
    score = (histogram - mean_mahalanobis) @ inv_covariance_mahalanobis @ (histogram - mean_mahalanobis)
    # ### OOD score ###

    return score


def min_max_scale_q(image: np.ndarray, q_min: int = 1, q_max: int = 99):
    image = np.clip(image, *np.percentile(np.float32(image), [q_min, q_max]))
    min_val, max_val = image.min(), image.max()
    return np.array((image.astype(np.float32) - min_val) / (max_val - min_val), dtype=image.dtype)


def min_max_scale(a: np.ndarray):
    a = np.copy(a)
    a -= a.min()
    a /= min(a.max(), 800.)
    a = np.clip(a, 0, 1)
    return a


def zoom_image(image, old_spacing, new_spacing):
    if not isinstance(new_spacing, (tuple, list, np.ndarray)):
        new_spacing = np.broadcast_to(new_spacing, 3)
    scale_factor = np.nan_to_num(np.float32(old_spacing) / np.float32(new_spacing), nan=1)
    return np.array(zoom(image.astype(np.float32), scale_factor, order=1))


def save_nb_zero_like(fname, path_inp, path_out):
    nii = nb.load(path_inp / fname)
    nb.save(nb.Nifti1Image(np.zeros_like(nii.get_fdata()), affine=nii.affine), path_out / fname)


def load_embeddings(data: str):
    path_embeddings = Path('/workspace/embeddings/')
    embeddings = np.load(path_embeddings / f'{data}_embeddings.npy')

    nan_rows = np.sum(np.isnan(embeddings), axis=1).nonzero()[0]
    embeddings = np.delete(embeddings, nan_rows, axis=0)

    return embeddings


def compute_wavelet_features(volume):
    # Perform a multi-level, 3D wavelet decomposition using the Daubechies 4 wavelet
    coeffs = pywt.wavedecn(volume, wavelet='db4', level=4)

    features = []

    # First handle the approximation coefficients (cAn)
    approximation_coeff = coeffs[0]
    mean = np.mean(approximation_coeff)
    variance = np.var(approximation_coeff)
    skewness = scipy.stats.skew(approximation_coeff.ravel())
    kurtosis = scipy.stats.kurtosis(approximation_coeff.ravel())
    features.extend([mean, variance, skewness, kurtosis])

    # Handle detail coefficients
    for details in coeffs[1:]:
        for key, coeff in details.items():
            mean = np.mean(coeff)
            variance = np.var(coeff)
            skewness = scipy.stats.skew(coeff.ravel())
            kurtosis = scipy.stats.kurtosis(coeff.ravel())
            energy = np.sum(coeff.ravel() ** 2)
            range_ = np.ptp(coeff.ravel())
            mad = np.mean(np.abs(coeff - mean))
            features.extend([mean, variance, skewness, kurtosis, energy, range_, mad])

    return np.array(features)


def get_histogram_for_file(args):
    fname, path_inp, data, n_bins = args
    return compute_histogram(load_and_prep_nifti(fname, path_inp, data), n_bins=n_bins)


def get_score_for_file(args):
    fname, path_inp, data, embeddings, n_bins = args
    return get_ood_score(fname, path_inp, data, embeddings, n_bins=n_bins)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inp", required=True, type=str)
    parser.add_argument("-o", "--out", required=True, type=str)

    parser.add_argument('-m', "--mode", type=str, default='sample', choices=('pixel', 'sample'), required=False)
    parser.add_argument('-d', '--data', type=str, choices=('brain', 'abdom'), required=True)

    parser.add_argument('--create_embeddings', action='store_true', required=False, default=False)

    parser.add_argument('--n_bins', type=int, default=150, required=False)

    args = parser.parse_known_args()[0]

    path_inp = Path(args.inp)
    path_out = Path(args.out)
    data = args.data

    fnames = os.listdir(path_inp)

    with Pool(cores) as pool:
        if args.create_embeddings:
            params = [(fname, path_inp, data, args.n_bins) for fname in fnames]
            embeddings = list(tqdm(pool.imap(get_histogram_for_file, params), total=len(params)))
            embeddings = np.stack(embeddings)
            np.save(f'{args.data}_embeddings.npy', np.float32(embeddings))
        else:
            if args.mode == 'pixel':
                for fname in fnames:
                    save_nb_zero_like(fname, path_inp, path_out)
            else:  # mode == 'sample':
                embeddings = load_embeddings(args.data)

                params = [(fname, path_inp, args.data, embeddings, args.n_bins) for fname in fnames]
                scores = list(tqdm(pool.imap(get_score_for_file, params), total=len(params)))
                scores = min_max_scale(np.float32(scores))
                for score, fname in zip(scores, fnames):
                    print(f'File: {fname}, score: {np.round(score, 2)}')
                    with open(str(path_out / (fname + '.txt')), 'w') as wf:
                        wf.write(str(score))


if __name__ == '__main__':
    main()
