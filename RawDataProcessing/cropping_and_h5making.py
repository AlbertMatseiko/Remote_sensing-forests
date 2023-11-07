# Importing
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
import h5py as h5
import cv2
from osgeo import gdal, osr
import tarfile
import glob


# Get coordinates from TIF image in its coordinate system
def get_coords_from_tif(path):
    # Get the existing coordinate system
    ds = gdal.Open(path)

    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    dx = gt[1]
    dy = -gt[5]

    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]

    # maxx = gt[0] + width*gt[1] + height*gt[2]
    # maxy = gt[3]
    centerx = gt[0] + width / 2 * gt[1] + height / 2 * gt[2]
    centery = gt[3] + width / 2 * gt[4] + height / 2 * gt[5]
    center = (centerx, centery)
    sizes = (width * dx, height * dy)
    return center, sizes


# Convert TIF coordinates into lattitude and longitude
def transorm_point_to_latlong(x, y, path):
    ds = gdal.Open(path)

    # projective coordinate system
    old_cs_config = """
    PROJCS["WGS 84 / UTM zone 39N",
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]],
        PROJECTION["Transverse_Mercator"],
        PARAMETER["latitude_of_origin",0],
        PARAMETER["central_meridian",51],
        PARAMETER["scale_factor",0.9996],
        PARAMETER["false_easting",500000],
        PARAMETER["false_northing",0],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["EPSG","32639"]]
    """
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(old_cs_config)

    # create the new coordinate system
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    # get the coordinates in lat long
    latlong = transform.TransformPoint(x, y)
    return latlong


# open and extract relevant scene's images in cache directory
def extract_scene_to_cache(path_to_data, s):
    path = path_to_data + '/' + s
    with tarfile.open(path) as tar:
        files = glob.glob('../../DATA/cache/*')
        for f in files:
            os.chmod(f, 0o777)
            os.remove(f)
        # data_list = os.listdir(tar)
        for i, m in enumerate(tar.getmembers()):
            if m.name.endswith('.TIF') and 'SR_B' in m.name:
                tar.extract(m, '../DATA/cache/')


# calculate the angle of image rotation (to get rid of black triangles)
def get_angle(image_npy):
    x1_max = 0
    x2_max = 0
    for b in range(7):
        y1 = image_npy.shape[1] - 1000
        x1 = np.nonzero(image_npy[:, y1, b])[0][0]
        if x1 > x1_max:
            x1_max = x1
        y2 = image_npy.shape[1] - 5000
        x2 = np.nonzero(image_npy[:, y2, b])[0][0]
        if x2 > x2_max:
            x2_max = x2
    tan = (y1 - y2) / (x1_max - x2_max)
    phi_rad = np.pi / 2 - np.arctan(tan)
    return phi_rad


# crop image on some delta pixels from edges after the rotation
def crop_black(new_npy, delta=200):
    x_min = np.nonzero(new_npy[:, new_npy.shape[1] - 5000, 0])[0][0] + delta
    x_max = np.nonzero(new_npy[:, new_npy.shape[1] - 5000, 0])[0][-1] - delta
    y_min = np.nonzero(new_npy[5000, :, 0])[0][0] + delta
    y_max = np.nonzero(new_npy[5000, :, 0])[0][-1] - delta
    return new_npy[x_min:x_max, y_min:y_max, :]


# Class that works with scene (TIF images) as a numpy array
class numpy_image():
    def __init__(self, s_name, path_to_data):

        # name of current scene
        self.s = s_name

        # extracting images of scene from .tar to cache
        extract_scene_to_cache(path_to_data, self.s)

        # path to extracted scene
        self.path = '../DATA/cache/'

        # collecting cached tifs into one numpy array
        name_tifs = sorted(os.listdir(self.path))
        scene = []
        for nt in name_tifs:
            path_to_tif = self.path + nt
            # z print(path_to_tif)
            band = cv2.imread(path_to_tif)[:, :, 0]
            scene.append(band)
        self.npy = np.array(scene).transpose(1, 2, 0)

        # calculating angle to rotate
        self.phi_rad = get_angle(self.npy)
        self.phi = self.phi_rad / np.pi * 180

        # rotation of image and cropping black boundaries
        self.npy = rotate(self.npy, self.phi)
        self.npy = crop_black(self.npy)

        # calculating eventual transformation in terms of affine trasfrom
        self.center, self.sizes = get_coords_from_tif(path_to_tif)
        w = self.sizes[0]  # size along x1 axes
        h = self.sizes[1]  # size along y1 axes
        self.a = (-h * np.cos(self.phi_rad) + w * np.sin(self.phi_rad)) / (
                    np.sin(self.phi_rad) ** 2 - np.cos(self.phi_rad) ** 2)  # size along x2 axes
        self.b = (h * np.sin(self.phi_rad) - w * np.cos(self.phi_rad)) / (
                    np.sin(self.phi_rad) ** 2 - np.cos(self.phi_rad) ** 2)  # size along y2 axes

        (x2_pixels, y2_pixels) = self.npy[:, :, 0].shape
        (self.dx2, self.dy2) = (self.a / x2_pixels, self.b / y2_pixels)

    # get actual coordinated of a pixel
    def get_geo_coords(self, x2, y2):
        x1 = self.center[0] + x2 * np.cos(self.phi_rad) + y2 * np.sin(self.phi_rad)
        y1 = self.center[1] - x2 * np.sin(self.phi_rad) + y2 * np.cos(self.phi_rad)
        return x1, y1

    # getting batch of small images from the initial one, concat with coordinates info
    def make_cropped_batch(self, size=512):
        Shape = self.npy.shape
        y_num = Shape[0] // size
        x_num = Shape[1] // size
        img_list = []

        x2_up_left = -self.b / 2
        y2_up_left = self.a / 2
        coords_list = []

        for i in range(y_num):
            for j in range(x_num):
                img = self.npy[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
                img_list.append(img)
                x2 = x2_up_left + self.dx2 * size * j
                y2 = y2_up_left - self.dy2 * size * i
                x_u_l, y_u_l = self.get_geo_coords(x2, y2)
                coords_list.append([x_u_l, y_u_l])
        return np.array(img_list), np.array(coords_list)


### Path to raw data
path_to_data = '../DATA/data_raw/'
scenes = sorted(os.listdir(path_to_data))

### Creating dir for h5 files if necessary
try:
    os.mkdir('../DATA/h5_files/')
except:
    pass

### Set desireble size of small images
size = 256

### Set True if you want to make a small dataset
SMALL_DATASET = False
if SMALL_DATASET == True:
    # 1/10 of all scenes will be added to dataset
    scenes = scenes[0:len(scenes) // 10]
    h5_name = 'LC08_L2SP_02_T1_' + str(size) + '_SMALL.h5'
else:
    h5_name = 'LC08_L2SP_02_T1_' + str(size) + '.h5'

### Path where to create h5 files
path_to_h5 = '../DATA/h5_files/' + h5_name

### Creates an unnormed h5 dataset
with h5.File(path_to_h5, 'w') as f:
    ### Iterting over .tar files
    for n, s in enumerate(scenes):

        ### Converting scene to numpy image
        image = numpy_image(s, path_to_data)

        ### Making batch of small images SIZExSIZExCHANNELS and their coords
        batch, coords = image.make_cropped_batch(size)

        ### Adding the batch and coords to h5 dataset
        if n == 0:
            f.create_dataset('all/data_raw', data=batch, maxshape=(None, size, size, 7))
            f.create_dataset('all/geo_coords', data=coords, maxshape=(None, 2))
            print("Initial dataset for the 1'st scene is created")
        else:
            f['all/data_raw'].resize((f['all/data_raw'].shape[0] + batch.shape[0]), axis=0)
            f['all/data_raw'][-batch.shape[0]:] = batch
            f['all/geo_coords'].resize((f['all/geo_coords'].shape[0] + coords.shape[0]), axis=0)
            f['all/geo_coords'][-coords.shape[0]:] = coords
            print(str(n + 1) + "'th scene is added")

### Calculating MEAN and VARIANCE for each channel ###

# Path to h5 file that needed to be normalized
path_to_h5_to_norm = '../DATA/h5_files/' + h5_name
with h5.File(path_to_h5_to_norm, 'r+') as f:
    # Initialize sum and sum of square variances for each channel
    SUM = 0
    SUM_SQ = 0

    # Number of small images in h5 file
    N = f['all/data_raw'].shape[0]
    print(N)

    # Size of every image in h5 file
    size = f['all/data_raw'].shape[1]

    # Set step to iterate over the file
    dN = 50

    # Calculating SUM and MEAN for each channel
    for i in range(0, N - dN, dN):
        SUM += np.sum(f['all/data_raw'][i:i + dN], axis=(0, 1, 2))
        if i % 100 == 0: print('Step number is:', i)
        i_last = i + dN
    SUM += np.sum(f['all/data_raw'][i_last:], axis=(0, 1, 2))
    MEAN = np.array(SUM, dtype='float64') / (N * size ** 2)

    # Calculating SUM_SQ and dispersion for each channel
    for i in range(0, N - dN, dN):
        SUM_SQ += np.sum((f['all/data_raw'][i:i + dN] - MEAN) ** 2, axis=(0, 1, 2))
        if i % 100 == 0: print('Step number is:', i)
        i_last = i + dN
    SUM_SQ += np.sum((f['all/data_raw'][i_last:] - MEAN) ** 2, axis=(0, 1, 2))
    DISP = SUM_SQ / (N * size ** 2)
    SIGMA = np.sqrt(DISP)

    # Adding info about mean and dispersion in the h5 dataset
    try:
        del f['all/norm_params']
    except:
        pass
    f.create_dataset('all/norm_params/mean_values', data=MEAN, maxshape=(7,))
    f.create_dataset('all/norm_params/sigma_values', data=SIGMA, maxshape=(7,))

### Replacing dataset with normilized one ###

# Creating datasets with normilized data
with h5.File(path_to_h5_to_norm, 'r+') as f:
    try:
        del f['all/data_norm']
    except:
        pass

    # Get mean and variance
    MEAN = f['all/norm_params/mean_values']
    SIGMA = f['all/norm_params/sigma_values']

    # Get number of small images
    N = f['all/data_raw'].shape[0]
    size = f['all/data_raw'].shape[1]

    # Set step to iterate over the file
    bs = 100

    # Normalizing data iterating over file, adding to data_norm dataset
    for i in range(0, N - bs, bs):
        batch = np.array(f['all/data_raw'][i:i + bs], dtype='float64')
        batch_norm = np.array((batch - MEAN) / SIGMA, dtype='float32')
        if i == 0:
            f.create_dataset('all/data_norm', data=batch_norm, maxshape=(None, size, size, 7))
        else:
            f['all/data_norm'].resize((f['all/data_norm'].shape[0] + bs), axis=0)
            f['all/data_norm'][-bs:] = batch_norm
        print('Batch number is: ', i)
        i_last = i + bs

    # Normilizing remainder
    batch = np.array(f['all/data_raw'][i_last:], dtype='float64')
    batch_norm = np.array((batch - MEAN) / SIGMA, dtype='float32')
    if i_last == 0:
        f.create_dataset('all/data_norm', data=batch_norm, maxshape=(None, size, size, 7))
    else:
        f['all/data_norm'].resize((f['all/data_norm'].shape[0] + batch_norm.shape[0]), axis=0)
        f['all/data_norm'][-batch_norm.shape[0]:] = batch_norm
    print('h5 file is normilized!')

    # Delete raw dataset, leaving normalized one
    del f['all/data_raw']

    print('Now keys of the file are:', f['all'].keys())
