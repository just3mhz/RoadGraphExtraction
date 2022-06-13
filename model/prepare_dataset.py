import os
import json
import argparse
import logging
import subprocess
import pickle

from collections import defaultdict

import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

from geometry import Transform


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


def make_mask(gdf, tile_path: str, output_path: str, burn_value: int = 255):
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


def make_graph(tile, geojson):
    raster_src = gdal.Open(tile)
    
    #  Get top-left and bottom-right corners 
    lon0, xres, xskew, lat0, yskew, yres = raster_src.GetGeoTransform()
    lon1 = lon0 + (raster_src.RasterXSize * xres)
    lat1 = lat0 + (raster_src.RasterYSize * yres)

    #  Transform to image rect coords
    transform = Transform(lon0, lat0, lon1, lat1, raster_src.RasterXSize, raster_src.RasterYSize)

    graph = defaultdict(set)
    features = json.load(open(geojson, 'r'))['features']
    for feature in features:
        geometry = feature['geometry']
        if geometry['type'] == 'LineString':
            coordinates = np.array(geometry['coordinates']).T
            add_line_to_graph(graph, coordinates, transform)
        if geometry['type'] == 'MultiLineString':
            for line in geometry['coordinates']:
                add_line_to_graph(graph, np.array(line).T, transform)
    return graph


def add_line_to_graph(graph, coordinates, transform):
    x, y = transform(coordinates[0], coordinates[1])
    n = x.shape[0]
    for i in range(1, n):
        graph[(x[i - 1], y[i - 1])].add((x[i], y[i]))
        graph[(x[i], y[i])].add((x[i - 1], y[i - 1]))


def make_data(tile_path, geojson_path, record):
    convert_to_8_bit(tile_path, record['tile'])
    make_mask(create_buffer_geopandas(geojson_path, 2, 6), tile_path, record['mask'])
    pickle.dump(make_graph(tile_path, geojson_path), open(record['graph'], 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--geojsons-dir', required=True)
    parser.add_argument('--city', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, args.city)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    records = []
    for i, image in enumerate(os.listdir(args.images_dir)):

        base_name = image.split('.')[0]
        geojson = base_name + '.geojson'
        try:
            record = {
                'tile': os.path.abspath(os.path.join(output_dir, f'tile_{i}_8bit.tif')),
                'mask': os.path.abspath(os.path.join(output_dir, f'tile_{i}_mask.tif')),
                'graph': os.path.abspath(os.path.join(output_dir, f'tile_{i}_graph.p')),
                'city': args.city
            }
            make_data(os.path.join(args.images_dir, image), os.path.join(args.geojsons_dir, geojson), record)
            records.append(record)
        except Exception as e:
            logging.warning(f"Exception while processing {image} | {geojson} | {e}")
            pass

        if (i + 1) % 100 == 0:
            logging.info(f'{i + 1} records processed')

    pd.DataFrame(records).to_csv(os.path.join(args.output_dir, f'tiles_{args.city}.csv'))
