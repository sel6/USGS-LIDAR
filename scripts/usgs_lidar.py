import matplotlib.image as mp
import urllib3
import json
import re
import numpy as np
import pdal
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, mapping
import logging
from logging.handlers import TimedRotatingFileHandler

class UsgsLidar:
    """
    A class that load, fetch, visualise, and transform publicly available LIDAR data on AWS.
    """
    
    def __init__(self, path = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/", pipeline_json_path: str="../pipeline.json") -> None:
        
        """
        Args:
            path (str, optional): url path location of the Lidar data. 
		Defaults to "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/"
            pipeline_json_path (str, optional): the json file with the pipeline structure. 
		Defaults to "../pipeline.json".
            
        Returns:
            [None]: nonetype object.
        """
            
        
        logging.basicConfig(filename="../logs/keep_track.log", level=logging.INFO, format="time: %(asctime)s, function: %(funcName)s, module: %(name)s, message: %(message)s \n")
            
        self.path = path
        self.txt = self.read_txt("../data/filenames.txt")
        self.a = self.read_json("../pipeline.json")
        self.metadata = self.read_csv("../data/metadata.csv")
    
    def read_json(self, json_path: str):
        """
        A method to read a json file
        
        Args:
            json_path (str): the location of the json file.
        """
        
        try:
            with open(json_path) as js:
                json_obj = json.load(js)
                logging.info("read json file successfully!")
            return json_obj

        except FileNotFoundError:
            logging.exception('File not found.')
        
    def fetch_polygon_boundaries(self, polygon: Polygon) -> tuple:
        """
        A method that fetch the polygon boundaries based on the input polygon
        
        Args:
            polygon (Polygon): the input polygon
            
        Returns:
            [tuple]: bounds and polygon exterior coordinates string.
        """
        polygon_df = gpd.GeoDataFrame([polygon], columns=['geometry'])

        polygon_df.set_crs(epsg=4326, inplace=True)
        polygon_df['geometry'] = polygon_df['geometry'].to_crs(epsg=3857)
        minx, miny, maxx, maxy = polygon_df['geometry'][0].bounds

        polygon_input = 'POLYGON(('
        xcords, ycords = polygon_df['geometry'][0].exterior.coords.xy
        for x, y in zip(list(xcords), list(ycords)):
            polygon_input += f'{x} {y}, '
        polygon_input = polygon_input[:-2]
        polygon_input += '))'
        
        logging.info("successfully fetched polygon boundaries")
        
        return f"({[minx, maxx]},{[miny,maxy]})", polygon_input
    
    def read_csv(self, csv_path, missing_values=["n/a", "na", "undefined"]) -> pd.DataFrame:
        """
        A method to read a csv file
        
        Args:
            csv_path (string): the location of the csv file.
            missing_values(string, optional): null expressions.
            
        Returns:
            [pandas.DataFrame]: pandas dataframe
        """
        try:
            df = pd.read_csv(csv_path, na_values=missing_values)
            logging.info("read csv successfully!")
                         
            return df

        except FileNotFoundError:
                         
            logging.exception("failed to load file!")             
            print('File not found.')
            
            
    def fetch_pipeline (self, region: str, polygon: Polygon) -> pdal.Pipeline:
        """
        A method to fill the empty values in the json pipeline and create pdal pipeline object
        
        Args:
            region (str): the filename of the region.
            polygon: (Polygon): the input polygon.
            
        Returns:
            [pdal.pipeline]: pdal pipeline object.     
        """
        url = f"{self.path}{region}/ept.json"
        boundary, poly = self.fetch_polygon_boundaries(polygon)
        
        self.a['pipeline'][0]['filename']= f"{self.path}{region}/ept.json"
        self.a['pipeline'][0]['polygon'] = poly
        self.a['pipeline'][0]['bounds'] = boundary
        pipeline = pdal.Pipeline(json.dumps(self.a))
        
        logging.info("loaded pipe successfully!")
        return pipeline
    
    def execute_pipeline(self, polygon: Polygon, epsg=4326, region: str = "IA_FullState") -> None:
        """
        A method to execute a pipeline and fetch data.
        
        Args:
            polygon (Polygon): A polygon object.
            epsg (int, optional): EPSG coordinate system. Default to 4326.
            region (str, optional): the filename of the region. Default to IA_FullState.
        
        Returns:
            [None]: nonetype object.
        """
        
        pipeline = self.fetch_pipeline(region, polygon)

        try:
            pipeline.execute()
            logging.info("pipeline executed successfully!")
                              
            return pipeline
        except RuntimeError as e:
            logging.exception('Pipeline execution failed')
            print(e)
    
    
    def create_gpd_df(self, epsg, pipe) -> gpd.GeoDataFrame:
        """
        A method to create geopandas dataframe from a pipeline object
        
        Args:
            epsg (int, optional): EPSG coordinate system.
            pipe (pdal.Pipeline): pipeline object.
            
        Returns:
                [Geopandas.GeoDataFrame]: a geopandas dataframe.
        """    
        try:
            cloud = []
            elevations =[]
            geometry=[]
            for row in pipe.arrays[0]:
                lst = row.tolist()[-3:]
                cloud.append(lst)
                elevations.append(lst[2])
                point = Point(lst[0], lst[1])
                geometry.append(point)
            gpd_df = gpd.GeoDataFrame(columns=["elevation", "geometry"])
            gpd_df['elevation'] = elevations
            gpd_df['geometry'] = geometry
            gpd_df = gpd_df.set_geometry("geometry")
            gpd_df.set_crs(epsg = epsg, inplace=True)
                              
            logging.info("created geopandas dataframe successfully!")
                              
            return gpd_df
        except RuntimeError as e:
            logging.exception("failed to create geopandas")
            print(e)

    def fetch_region_data(self, polygon: Polygon, epsg=4326) -> gpd.GeoDataFrame:
        """
        A method to fetch the data of a region.
        
        Args:
            polygon (polygon): a polygon object.
            epsg (int, optional): EPSG coordinate system.
            
        Returns:
            [Geopandas.GeoDataFrame]: a geopandas dataframe.
        """    
        pipeline = self.execute_pipeline(polygon, epsg)
        logging.info("fetched region data successfully!")
                              
        return self.create_gpd_df(epsg, pipeline)
    
    def read_txt(self, txt_path: str) -> list:
        """
        A method to read text file.
        
        Args:
            txt_path (str): path to the text file.
        Returns:
            [list]: list of text files.
        """
        try:
            with open(txt_path, "r") as f:
                text_file = f.read().splitlines()
                logging.info("read text file successfully!")
                              
            return text_file

        except Exception as e:
            logging.exception("failed to load text file")
            print(e)
            
    def fetch_name_and_year(self, location: str) -> tuple:
        """
        A method to fetch name and year from file name.
        
        Args:
            location (str): location of file.
        
        Returns:
            [tuple]: tuple of name and year.
        """
        location = location.replace('/', '')
        regex = '20[0-9][0-9]+'
        match = re.search(regex, location)
        if(match):
          logging.info("there is a name and year match!")
          return (location[:match.start() - 1], location[match.start():match.end()])
        else:
          logging.info("there is no match!")
          return (location, None)
    
   
    def fetch_metadata(self) -> pd.DataFrame:
        """
        A method to create metadata for EPT files available on AWS.
        
        Returns:
            [pandas.DataFrame]: dataframe of the metadata.
        """
    
        metadata = pd.DataFrame(columns=['filename', 'region',
                          'year', 'xmin', 'xmax', 'ymin', 'ymax', 'points'])

        index = 0
        for lists in self.txt:
          r = urllib3.PoolManager().request('GET', self.path + lists + "ept.json")
          if r.status == 200:
            j = json.loads(r.data)
            region, year = self.fetch_name_and_year(lists)

            metadata = metadata.append({
                'filename': lists.replace('/', ''),
                'region': region,
                'year': year,
                'xmin': j['bounds'][0],
                'xmax': j['bounds'][3],
                'ymin': j['bounds'][1],
                'ymax': j['bounds'][4],
                'points': j['points']}, ignore_index=True)

            metadata.to_csv("../data/metadata.csv")
        
        logging.info("returned metadata successfully!")
        return(metadata)
    
    
    def fetch_regions(self, polygon: Polygon, epsg=4326) -> list:
        """
        A method to fetch region(s) within a polygon.
        
        Args:
            polygon (Polygon): a polygon object.
            epsg (int, optional): EPSG coordinate system.
            
        Returns:
            [list]: lists of regions within the polygon.
        """
    
        polygon_df = gpd.GeoDataFrame([polygon], columns=['geometry'])

        polygon_df.set_crs(epsg, inplace=True)
        polygon_df['geometry'] = polygon_df['geometry'].to_crs(epsg=3857)
        minx, miny, maxx, maxy = polygon_df['geometry'][0].bounds

        cond_xmin = self.metadata.xmin <= minx
        cond_xmax = self.metadata.xmax >= maxx
        cond_ymin = self.metadata.ymin <= miny
        cond_ymax = self.metadata.ymax >= maxy


        df = self.metadata[cond_xmin & cond_xmax & cond_ymin & cond_ymax]
        sort_df = df.sort_values(by=['year'])
        regions = sort_df['filename'].to_list()
        if(len(df)==0):
            print("polygon is not located")
        
        logging.info("fetched region successfully!")
        return regions   
    
    def fetch_data(self, polygon: Polygon, region="IA_FullState") -> dict:
        """
        A method to fetch the data of a region.
        
        Args:
            polygon (Polygon): a polygon object.
            region (str, optional): the region where the data will be extracted from.
            
        Returns:
            [dict]: a dictionary object with year, geopandas dataframe pair.
        """
        regions = self.fetch_regions(polygon)

        region_dicto = {}
        for i in regions:
            if i==region:
                year = (self.metadata[self.metadata.filename == i].year.values[0])
                year = str(year)
                
                if(year=="nan"):
                    year = "Year: not_specified"
                
                region_df = self.fetch_region_data(polygon)
            
                if region_df.empty == False:
                    region_dicto[year] = region_df
        
        logging.info("fetched data successfully")
        return(region_dicto)
    
    def plot_terrain(self, gdf: gpd.GeoDataFrame, fig_size: tuple=(12, 10), size: float=0.01) -> None:
        """
        A method to plot points in geopandas dataframe as a 3D scatter plot.
        
        Args:
            gdf (GeoDataFrame): a geopandas dataframe containing columns of elevation and geometry.
            fig_size (tuple, optional): filesze of the figure to be displayed. Defaults to (12, 10).
            size (float, optional): size of the points to be plotted. Defaults to 0.01.
        
        Returns:
            [None]: nonetype object.
        """
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax = plt.axes(projection='3d')
        ax.scatter(gdf.geometry.x, gdf.geometry.y, gdf.elevation, s=size)
        plt.show()
        
        logging.info("plotted terrain successfully!")
                              
    def save_heatmap(self, df:gpd.GeoDataFrame, png_path:str, title:str) -> None:
        """
        A method to plot and save a heatmap.
        
        Args:
            df (GeoDataFrame): a geopandas dataframe containing columns of elevation and geometry.
            png_path (str): the path to save the heatmap as PNG.
            title (str): the tite  of the image.
        Returns:
            [None]: nonetype object.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        df.plot(column='elevation', ax=ax, legend=True, cmap="terrain")
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(png_path, dpi=120)
        plt.axis('off')
        plt.close()
        
        logging.info("saved hitmap successfully!")
        
    def load_heatmap(self, png_path:str) -> None:
        """
        A method to load a saved image.
        
        Arg:
            png_path (str): the path of the image to load.
        Returns:
            [None]: nonetype object.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        img = mp.imread(png_path)
        imgplot = plt.imshow(img)
        plt.axis('off')
        plt.show() 
        
        logging.info("loaded hitmap successfully!")
        
    def subsample(self, gdf: gpd.GeoDataFrame, res: int = 3) -> gpd.GeoDataFrame:
        """
        A method to sample a point cloud data by implementing a decimation and voxel grid sampling to reduce point cloud data density.

        Args:
            gdf (gpd.GeoDataFrame): a geopandas dataframe containing columns of elevation and geometry.
            res (int, optional): resolution. Defaults to 3.

        Returns:
            [Geopandas.GeoDataFrame]: a geopandas dataframe.
        """

        points = np.vstack((gdf.geometry.x, gdf.geometry.y, gdf.elevation)).transpose()

        voxel_size=res

        non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
        idx_pts_vox_sorted=np.argsort(inverse)

        voxel_grid={}
        grid_barycenter=[]
        last_seen=0

        for idx,vox in enumerate(non_empty_voxel_keys):
            voxel_grid[tuple(vox)]= points[idx_pts_vox_sorted[
            last_seen:last_seen+nb_pts_per_voxel[idx]]]
            grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
            last_seen+=nb_pts_per_voxel[idx]

        sub_sampled =  np.array(grid_barycenter)
        df_subsampled = gpd.GeoDataFrame(columns=["elevation", "geometry"])

        geometry = [Point(x, y) for x, y in zip( sub_sampled[:, 0],  sub_sampled[:, 1])]

        df_subsampled['elevation'] = sub_sampled[:, 2]
        df_subsampled['geometry'] = geometry
        
        logging.info("subsampled cloud data successfully!")
        
        return df_subsampled
    
    def convert_epsg(self, df: gpd.GeoDataFrame, column: str, epsg_inp = 4326, epsg_out = 3857) -> gpd.GeoDataFrame:
        """
        A method that converts EPSG coordinate system

        Args:
            df (gpd.GeoDataFrame): a geopandas dataframe containing columns of elevation and geometry.
            column (str): the column geometry.
            epsg_inp (int): the current geometry EPSG type.
            epsg_out (int): EPSG type the geometry will be converted to.

        Returns:
                [Geopandas.GeoDataFrame]: a geopandas dataframe.    
        """
        try:
            df = df.set_crs(epsg_inp)
            df[column] = df[column].to_crs(epsg_out)
            df = df.set_crs(epsg_out)
            
            logging.info("successfully converted epsg format")
            
            return df
        except Exception as e:
            logging.exception(e)
                              
if __name__=="__main__":
    US = UsgsLidar()
    MINX, MINY, MAXX, MAXY = [-93.759055, 41.925015, -93.766155, 41.935015]
    polygon = Polygon(((MINX, MINY), (MINX, MAXY), (MAXX, MAXY), (MAXX, MINY), (MINX, MINY)))
    heatmap_path = "../data/heatmap2.png"
    output = US.fetch_data(polygon)
    terrain_view = US.plot_terrain(output, size=0.001)
    saved = US.save_heatmap(output, heatmap_path, "Polygon boundary map")
    loaded = US.load_heatmap(heatmap_path)
    print(output)
