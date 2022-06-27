from zipfile import ZipFile
from io import TextIOWrapper

import numpy as np
from skimage.filters import threshold_li
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects, ball, binary_erosion
from scipy.ndimage import median_filter
from sklearn.linear_model import LinearRegression


def parse_simulation_dat(file):

    with open(file) as fd:
        return parse_simulation(fd)

def parse_simulation_zip(file):

    with ZipFile(file) as _zip:
        with _zip.open(_zip.filelist[0]) as fd:
            return parse_simulation(TextIOWrapper(fd))

def parse_simulation(fd):
    '''
    read rasterized simulation results from file-like object
    '''

    lines = fd.readlines()

    # 'pixelsize shape' header line
    psz, shape = lines[0].split()
    psz, shape = float(psz), int(shape)

    # new files have second line containing a single number, skip
    # TODO: what is this?
    if not ' ' in lines[1]:
        remaining_lines = lines[2:]
    else:
        remaining_lines = lines[1:] 

    # read pixels, drop newlines, whitespace
    pixel_str = ''.join(remaining_lines).replace(' ', '').replace('\n', '')
    pixels = list(map(int, iter(pixel_str)))

    return psz, np.array(pixels).reshape((shape,) * 3)

def fwhm2sigma(fwhm):

    FACTOR = 2 * np.sqrt(2 * np.log(2))
    return fwhm / FACTOR

def segment_like_paper(patch, clahe_size=78, min_object_size=500, radius=2):
    '''
    Segmentation pipeline similar to ChromEMT pipeline
    clahe_size is in pixels, 78 ~ 100nm at 1.28nm pixel size
    '''

    # 1) CLAHE
    patch_eq = equalize_adapthist(patch, clahe_size)
    
    # 2) Li thresholding
    mask = patch_eq < threshold_li(patch_eq)

    # 3) ImaheJ "Remove Outliers..." should correspond to median filter
    # TODO: planewise matches ImageJ more
    mask = median_filter(mask, footprint=ball(radius))

    # 4) remove small objects should correspond to size threshold in 3D Object Counter
    mask = remove_small_objects(mask, min_object_size)
    return mask

def continuous_erosion(mask, erosion_radius=None):
    '''
    Continuous erosion to estimate chromatin thickness

    Parameters
    ==========
    mask: binary array of ndim == 3
        the mask to erode
    erosion_radius: iterable of floats/ints (optional)
        radii at which to calculate residual chromatin volume after erosion
        defaults to integer pixel radii up to half the z size of mask

    Returns
    =======
    residual_volumes: float array with shape == (len(erosion_radius),)
        residual volume fractions
    '''

    # default radii: integer pixels
    if erosion_radius is None:
        erosion_radius = np.arange(1, mask.shape[0]//2)

    # TODO: slow, parallelize or iterative erosion with iterative application of small sphere
    eroded = np.stack([binary_erosion(mask, ball(r)) for r in erosion_radius])

    return eroded.sum(axis=(1,2,3)) / mask.sum()

def linear_fit_to_residual_volume(residual_volume, erosion_radius=None, n_first_values_to_include=5):
    '''
    Fit linear model erosion_radius ~ residual volume to residual volumes
    Returns
    =======
    estimated_diameter: float
        the estimated diameter (2x erosion_radius intercept / intersection of model with x-axis)
    model: LinearModel
        the fitted model
    '''

    # default radii: integer pixels
    if erosion_radius is None:
        erosion_radius = np.arange(1, residual_volume.size+1)

    residual_volume = residual_volume[:n_first_values_to_include]
    erosion_radius = erosion_radius[:n_first_values_to_include]

    lm = LinearRegression()
    lm.fit(residual_volume.reshape(-1, 1), erosion_radius)
    
    return lm.intercept_ * 2, lm