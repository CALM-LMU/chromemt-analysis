from zipfile import ZipFile
from io import TextIOWrapper

import numpy as np
from skimage.filters import threshold_li
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects, ball, binary_erosion
from scipy.ndimage import median_filter
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LinearRegression

def disk_centered(radius, dtype=np.uint8):
    """Generates a flat, disk-shaped footprint.
    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.
    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.
    Other Parameters
    ----------------
    dtype : data-type
        The data type of the footprint.
    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    radius_ceil = int(np.ceil(radius))
    L = np.arange(-radius_ceil, radius_ceil + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)

def parse_simulation_dat(file):

    with open(file) as fd:
        return parse_simulation(fd)

def parse_simulation_zip(file, subfile=None):

    with ZipFile(file) as _zip:
        
        if subfile is None:
            f = _zip.filelist[0]
        else:
            try:
                f = next((f for f in _zip.filelist if f.filename == subfile))
            except StopIteration:
                raise ValueError(f'file {subfile} not found in zip archive {file}. Available files in zip:\n* ' +
                                '\n* '.join([f.filename for f in _zip.filelist]))
        with _zip.open(f) as fd:
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

def segment_like_paper(patch, clahe_size=78, min_object_size=500, radius=2, planewise_median=True, planewise_clahe=True, imagej_outlier_bright=True, imagej_outlier_dark=True):
    '''
    Segmentation pipeline similar to ChromEMT pipeline
    clahe_size is in pixels, 78 ~ 100nm at 1.28nm pixel size
    '''

    # normailzed clip limit? TODO: is this equivalent to 3 in ImageJ?
    cl = 3 / 256

    # 1) CLAHE
    if planewise_clahe:
        patch_eq = np.stack([equalize_adapthist(patch_i, clahe_size, cl) for patch_i in patch])
    else:
        patch_eq = equalize_adapthist(patch, clahe_size, cl)
    
    # 2) Li thresholding
    mask = patch_eq < threshold_li(patch_eq)

    # 3) ImageJ "Remove Outliers..." should correspond to median filter
    # NOTE: planewise matches ImageJ more
    if planewise_median:
        for mask_i in mask:
            # NOTE: ImageJ seems to use odd-sized radius+0.5 selems
            # NOTE: we replicate doing ImageJ remove bright (in inverted LUT, so actually dark) and then remove dark
            if imagej_outlier_bright:
                mask_i |= median_filter(mask_i, footprint=disk_centered(radius+0.5))
            if imagej_outlier_dark:
                mask_i &= median_filter(mask_i, footprint=disk_centered(radius+0.5))
    else:
        mask = median_filter(mask, footprint=ball(radius))

    # 4) remove small objects should correspond to size threshold in 3D Object Counter
    mask = remove_small_objects(mask, min_object_size)
    return mask

def continuous_erosion_iterative(mask, n_iterations=10, connectivity=1):

    selem = generate_binary_structure(3, connectivity=connectivity)

    eroded = [binary_erosion(mask, selem)]
    for _ in range(n_iterations-1):
        eroded.append(binary_erosion(eroded[-1], selem))

    return np.array([e.sum()/mask.sum() for e in eroded])

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


def continuous_erosion_edt(mask, erosion_radius=None):
    '''
    Continuous erosion to estimate chromatin thickness
    Calculated via Euclidean distance tranform and measuring fraction of mask having d > erosion radius

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

    dt = distance_transform_edt(mask)

    # default radii: integer pixels
    if erosion_radius is None:
        erosion_radius = np.arange(1, mask.shape[0]//2)

    # residual volume: area of EDT > erosion radius
    eroded = np.array([(dt > r).sum() for r in erosion_radius])

    return eroded / mask.sum()


def linear_fit_to_residual_volume2(residual_volume, erosion_radius=None, n_first_values_to_include=5):
    '''
    Fit linear model erosion_radius ~ residual volume to residual volumes ( x ~ y )
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

    # fit erosion_radius ~ residual_volume + 1
    # that way, intercept of fitted line := average radius
    lm = LinearRegression()
    lm.fit(residual_volume.reshape(-1, 1), erosion_radius)
    
    return lm.intercept_ * 2, lm


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

    # fit residual vol ~ erosion_radius + 1
    lm = LinearRegression()
    lm.fit(erosion_radius.reshape(-1, 1), residual_volume)
    
    # estimated radius: zero-crossing of fited line
    radius_estimated =  - lm.intercept_ / lm.coef_[0]

    return radius_estimated * 2, lm

def hypersphere(rank, radius):
    '''
    rank-n hypersphere, should be just like disk/ball
    '''
    return np.linalg.norm(np.stack(np.meshgrid(*[np.arange(-radius, radius+1)]*rank, indexing='ij')), axis=0)<=radius