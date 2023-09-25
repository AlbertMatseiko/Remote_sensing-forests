import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr

class VisualClass:
    def __init__(self, path_to_h5 = './data/LC08_L2SP_02_T1_cropped.h5'):
        self.path_to_h5 = path_to_h5

    def transform_point_to_latlong(self, x,y):

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
        old_cs .ImportFromWkt(old_cs_config)

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
        new_cs .ImportFromWkt(wgs84_wkt)


        # create a transform object to convert between coordinate systems
        transform = osr.CoordinateTransformation(old_cs,new_cs) 
        #get the coordinates in lat long
        latlong = transform.TransformPoint(x,y)
        return latlong

    def draw_result(self, predicted_classes, img_no):
        with h5py.File(self.path_to_h5, 'r') as f:
            MEAN = f['all/norm_params/mean_values'][:]
            SIGMA = f['all/norm_params/sigma_values'][:]
            GEO = f['all/geo_coords'][img_no]
            img_norm = f['all/data_norm'][img_no,:,:,:]
            img = (img_norm*SIGMA + MEAN) / 2**16

        def renorm(img, axis=(0, 1)):
            img_min = img.min(axis)
            img_max = img.max(axis)
            return (img - img_min) / (img_max - img_min)

        plt.figure(figsize=(8, 8))
        plt.suptitle('X = '+ str(np.round(GEO[0], 1))+'; Y = '+ str(np.round(GEO[1], 1)))

        plt.subplot(2, 2, 1)
        plt.imshow(img[:, :, 5:2:-1])

        plt.subplot(2, 2, 2)
        plt.imshow(predicted_classes[img_no])

    def get_rgb_image(self, img_no):
        with h5py.File(self.path_to_h5, 'r') as f:
            MEAN = f['all/norm_params/mean_values'][:]
            SIGMA = f['all/norm_params/sigma_values'][:]
            GEO = f['all/geo_coords'][img_no]
            img_norm = f['all/data_norm'][img_no,:,:,:]
        img = (img_norm*SIGMA + MEAN) / 2**16
        return img[:, :, 5:2:-1]

    def get_image(self, start, stop):
        with h5py.File(self.path_to_h5, 'r') as f:
            MEAN = f['all/norm_params/mean_values'][start:stop]
            SIGMA = f['all/norm_params/sigma_values'][start:stop]
            GEO = f['all/geo_coords'][start:stop]
            img_norm = f['all/data_norm'][start:stop,:,:,:]
        img = (img_norm*SIGMA + MEAN) / 2**16
        return img, GEO

    def get_norm_image(self, start, stop):
        with h5py.File(self.path_to_h5, 'r') as f:
            GEO = f['all/geo_coords'][start:stop]
            img_norm = f['all/data_norm'][start:stop,:,:,:]
        return img_norm, GEO

    def draw_layers(self, img_no, predicted_classes, CLASSES = 10, opacity=0.3):

        with h5py.File(self.path_to_h5, 'r') as f:
            GEO = f['all/geo_coords'][img_no]
        GEO = self.transform_point_to_latlong(*GEO)

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Original Image", "Classes", "Overlay", ""))
        rgb = self.get_rgb_image(img_no)

        fig.add_trace(px.imshow(rgb).data[0], 
                      row = 1,
                      col = 1)
        fig.add_trace(px.imshow(rgb).data[0], 
                      row = 1,
                      col = 2)
        fig.add_trace(px.imshow(rgb).data[0], 
                      row = 1,
                      col = 3)

        for i in range(CLASSES):
            class_mask = np.where(predicted_classes[img_no] == i, i, np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=class_mask, 
                    colorscale=[[0, f"hsv({i*360/20},100%,100%)"], [1, f"hsv({i*360/20},100%,100%)"]],
                    hoverongaps=False,
                    showscale=False,
                    showlegend=True,
                    name=f"Class {i}",
                    opacity=1
                ), row=1, col=2)
            fig.add_trace(
                go.Heatmap(
                    z=class_mask, 
                    colorscale=[[0, f"hsv({i*360/20},100%,100%)"], [1, f"hsv({i*360/20},100%,100%)"]],
                    hoverongaps=False,
                    showscale=False,
                    showlegend=False,
                    name=f"Class {i}",
                    opacity=opacity
                ), row=1, col=3)

        fig.update_layout(title_text="GEO COORDS: Lat = "+str(np.round(GEO[0], 5)) +' Long = '+str(np.round(GEO[1], 5)))

        return fig