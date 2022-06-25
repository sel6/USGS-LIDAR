# USGS-LIDAR

**Table of Content**

 -[Project Objective](#project objective)
 
 -[Data](#data)
 
 -[Requirments](#requirments)
 
 -[Tasks](#tasks)
 
 -[Install](#install)
 
## Project Objective
The task is to produce an easy to use, reliable and well designed python module that domain experts and data scientists can use to fetch, visualize, and transform publicly available satellite and LIDAR data. In particular, the code should interface with USGS 3DEP and fetch data using their API. 


## Data

The data can be found at amazon s3 bucket in https://s3-us-west-2.amazonaws.com/usgs-lidar-public.

## Tasks

* Data Fetching and Loading:
  * Write a python code that receives Field boundary polygon in geopandas dataframes as input and outputs a Python dictionary with all years of data           available and geopandas grid point file with elevations encoded in the requested CRS.

* Terrain Visualization:
  * Include an option to graphically display the returned elevation files as either a 3D render plot or as a heatmap. 

* Data Transformation 
  * Add a column in the geopandas dataframe with a returned (Topographic Wetness Index) TWI.
  * Write a python code that takes elevation points output from the USGS LIDAR tool and interpolates them to a grid.
    
    Inputs:
      * A single year geopandas elevation point dataframe returned from the tool above.
      * Desired output resolution (in meters).    
    
    Outputs:

     * An interpolated grid of points with interpolated elevation information
     * An option to visualize the output grid as a 3D render or heatmap to visually compare to the original, un-interpolated elevation data.

### Install

