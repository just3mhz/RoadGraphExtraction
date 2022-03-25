import os
import argparse
import logging
import subprocess

import geopandas as gpd
import numpy as np
import osmnx as ox
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

def convert_to_8_bit(input_file: str, output_file: str, output_pixel_type: str = "Byte", output_format: str = "GTiff"):
    percentiles = (2, 98)

    src = gdal.Open(input_file)
    cmd = ['gdal_translate', '-ot', output_pixel_type, '-of', output_format]

    for band_id in range(1, src.RasterCount + 1):
        band = src.GetRasterBand(band_id).ReadAsArray().flatten()
        bmin = np.percentile(band, percentiles[0])
        bmax = np.percentile(band, percentiles[1])
        cmd.extend([f'-scale_{band_id}', f'{bmin}', f'{bmax}', f'{0}', f'{255}'])
    
    cmd.extend([input_file, output_file])
    
    logging.info(f'Running cmd: {" ".join(cmd)}')
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def create_buffer_geopandas(geojson, distance, roundness):
    gdf = gpd.read_file(geojson)
    gdf['type'] = gdf['road_type'].values
    gdf['class'] = 'highway'
    gdf['highway'] = 'highway'

    if gdf.empty:
        return [], []

    gdf_utm = ox.project_gdf(gdf)
    gdf_utm['geometry'] = gdf_utm.buffer(distance, roundness)
    
    gdf_utm_dissolve = gdf_utm.dissolve(by='class')
    gdf_utm_dissolve.crs = gdf_utm.crs

    return gdf_utm_dissolve.to_crs(gdf.crs)


def make_mask(gdf, tile_path: str, output_path: str, burn_value: int = 150):
    tile = gdal.Open(tile_path)
    target = gdal.GetDriverByName('GTiff').Create(
            output_path,
            tile.RasterXSize,
            tile.RasterYSize,
            1,
            gdal.GDT_Byte)
    target.SetGeoTransform(tile.GetGeoTransform())

    raster = osr.SpatialReference()
    raster.ImportFromWkt(tile.GetProjectionRef())
    target.SetProjection(raster.ExportToWkt())

    band = target.GetRasterBand(1)
    band.SetNoDataValue(0)

    output_driver = ogr.GetDriverByName('MEMORY')
    output_data_source = output_driver.CreateDataSource('memData')
    output_layer = output_data_source.CreateLayer(
            'states_extent', raster, geom_type=ogr.wkbMultiPolygon)

    burn_field = 'burn'
    id_field = ogr.FieldDefn(burn_field, ogr.OFTInteger)
    output_layer.CreateField(id_field)
    feature_defn = output_layer.GetLayerDefn()

    for geometry_shape in gdf['geometry'].values:
        output_feature = ogr.Feature(feature_defn)
        output_feature.SetGeometry(ogr.CreateGeometryFromWkt(geometry_shape.wkt))
        output_feature.SetField(burn_field, burn_value)
        output_layer.CreateFeature(output_feature)

    gdal.RasterizeLayer(target, [1], output_layer, burn_values=[burn_value])


if __name__ == '__main__':
    convert_to_8_bit('input.tif', 'output.tif')
    make_mask(
            create_buffer_geopandas('input.geojson', 2, 6),
            'input.tif',
            'output_mask.tif')

