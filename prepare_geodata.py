import pandas as pd
import geopandas as gpd
import osmnx as ox
from termcolor import cprint
import math
from numpy import random
import os
import shapely
from data import *
import fiona


def get_geodata_from_place(place_name: str):
    """
    Get geodata from a place name using the OSMnx library. The name of the place must be a valid OSM place name. The data is filtered for buildings.
    :param place_name: str
    :return: A GeoDataFrame containing building data for the specified place
    :rtyoe: gpd.GeoDataFrame
    """
    tags = {'building': True} # Get only buildings
    cprint(f"Start: Retrieving geodata for {place_name}")
    # Get the geometry of the place
    try:
        gdf_raw_geo = ox.features_from_place(place_name, tags)
        try:
            # remove rows with "man_mande" not emtpy, if this column exists
            gdf_raw_geo = gdf_raw_geo[gdf_raw_geo['man_made'].isna()]
        except:
            pass
        #reset index                                              
        gdf_raw_geo.reset_index(inplace=True)
        cprint(f"Done: Geodata for {place_name} successfully retrieved.", "green")
        return gdf_raw_geo
    except Exception as e:
        cprint(f"Error: {e}", "red")
        return None

def get_geodata_from_polygon(polygon: shapely.geometry.polygon.Polygon):
    """
    Get geodata from a polygon using the OSMnx library. The data is filtered for buildings.
    :param polygon: shapely.geometry.polygon.Polygon
    :return: A GeoDataFrame containing building data for the specified place
    :rtype: gpd.GeoDataFrame
    """
    tags = {'building': True} # Get only buildings
    cprint(f"Start: Retrieving geodata for the {polygon}")
    # Get the geometry of the place
    try:
        gdf_raw_geo = ox.features_from_polygon(polygon, tags)
        try:
            # remove rows with "man_mande" not emtpy
            gdf_raw_geo = gdf_raw_geo[gdf_raw_geo['man_made'].isna()]
        except:
            pass
        #reset index 
        # set a numeric index
        gdf_raw_geo.reset_index(inplace=True)
        cprint(f"Done: Geodata for the polygon {polygon} successfully retrieved.", "green")
        return gdf_raw_geo
    except Exception as e:
        cprint(f"Error: {e}", "red")
        return None


def extract_relevat_data(gdf_in: gpd.GeoDataFrame):
    """
    Extract the relevant data from the raw geodata that are needed for further analysis. Data that are not available are set to None, and a warning is printed.
    :param gdf_in: GeoDataFrame with the raw geodata
    :return: GeoDataFrame containing the relevant data
    :rtype: gpd.GeoDataFrame
    """

    # extract the relevant data from the raw geodata that are needed for further analysis
    cprint(f"Start: Extracting relevant data from the raw geodata")
    # copy the gemetry coulumn
    gdf_out = gdf_in[['geometry']].copy()

    # add a unique id for each building
    gdf_out['building_id'] = 'building_' + gdf_in.index.astype(str)

    # add the addr: information if available
    if 'addr:city' in gdf_in.columns:
        gdf_out['addr:city'] = gdf_in['addr:city']
        cprint(f"Added addr:city", "green")
    else:
        gdf_out['addr:city'] = None
        cprint(f"Warning: addr:city not found in the raw data", "yellow")

    if 'addr:postcode' in gdf_in.columns:
        gdf_out['addr:postcode'] = gdf_in['addr:postcode']
        cprint(f"Added addr:postcode", "green")
    else:
        gdf_out['addr:postcode'] = None
        cprint(f"Warning: addr:postcode not found in the raw data", "yellow")

    if 'addr:street' in gdf_in.columns:
        gdf_out['addr:street'] = gdf_in['addr:street']
        cprint(f"Added addr:street", "green")
    else:
        gdf_out['addr:street'] = None
        cprint(f"Warning: addr:street not found in the raw data", "yellow")
    
    if 'addr:housenumber' in gdf_in.columns:
        gdf_out['addr:housenumber'] = gdf_in['addr:housenumber']
        cprint(f"Added addr:housenumber", "green")
    else:
        gdf_out['addr:housenumber'] = None
        cprint(f"Warning: addr:housenumber not found in the raw data", "yellow")
    
    # add building information if available (according to the OSM wiki)
    if 'building' in gdf_in.columns:
        gdf_out['building'] = gdf_in['building']
        cprint(f"Added building", "green")
    else:
        gdf_out['building'] = None
        cprint(f"Warning: building not found in the raw data", "yellow")

    if 'building:levels' in gdf_in.columns:
        gdf_out['building:levels'] = gdf_in['building:levels']
        cprint(f"Added building:levels", "green")   
    else:
        gdf_out['building:levels'] = None
        cprint(f"Warning: building:levels not found in the raw data", "yellow")

    if 'height' in gdf_in.columns:
        gdf_out['height'] = gdf_in['height']
        cprint(f"Added height", "green")
    else:
        gdf_out['height'] = None
        cprint(f"Warning: height not found in the raw data", "yellow")

    if 'building_flats' in gdf_in.columns:
        gdf_out['building_flats'] = gdf_in['building_flats']
        cprint(f"Added building_flats", "green")
    else:
        gdf_out['building_flats'] = None
        cprint(f"Warning: building_flats not found in the raw data", "yellow")

    if 'construction_date' in gdf_in.columns:
        gdf_out['construction_date'] = gdf_in['construction_date']
        cprint(f"Added construction_date", "green")
    else:
        gdf_out['construction_date'] = None
        cprint(f"Warning: construction_date not found in the raw data", "yellow")

    cprint(f"Done: Relevant data extracted from the raw geodata", "green")
    return gdf_out  

def estimate_data_osm(gdf_in: gpd.GeoDataFrame):
    """
    Estimate the required data for the subsequent analysis (e.g. number of floors) just from the osm data. Where not sufficient data is available, the data is estimated from the available data. 
    Note that the estimation is based on the OSM data and might not be accurate, depending on the quality of the OSM data.
    :param gdf_in: GeoDataFrame containing the relevant data
    :return: GeoDataFrame containing the estimated data
    :rtype: gpd.GeoDataFrame
    """
    cprint(f"Start: Processing data for building classification")
    gdf_out = gdf_in[['building_id','geometry']].copy()

    # get construction year
    gdf_out['year_of_construction'] = gdf_in['construction_date']
    gdf_out.loc[gdf_out['year_of_construction'].isna(), 'year_of_construction'] = -99 # set a numeric value for missing data eaysier handling
    data_avail_year_of_construction = 1 - gdf_in['construction_date'].isna().sum() / len(gdf_in)
    cprint(f"Data availability for year of construction: {data_avail_year_of_construction*100:.2f} %")

    # calculate the projected area of the building
    gdf_out['projected_ground_area'] = gdf_in['geometry'].to_crs(epsg=3035).area
    data_avail_area = (gdf_out['projected_ground_area'] > 0).sum() / len(gdf_out)
    cprint(f"Data availability for projected ground area: {data_avail_area*100:.2f} %")

    # get the number of floors 
    for index, row in gdf_in.iterrows():
        try:
            # try to get the actual number of floors first
            gdf_out.at[index, 'number_of_floors'] = int(row['building:levels']) 
        except:
            try:
                if math.isnan(row['height']) == False:
                    # as alternative estimate the number of buildings by the average height of a floor
                    gdf_out.at[index, 'number_of_floors'] = round(row['height'] / 3.5, 0)
                else:
                    # if no information is available, assume the height based on a average ground area to height ratio
                    gdf_out.at[index, 'number_of_floors'] = round(math.sqrt(gdf_out.at[index,'projected_ground_area']) / (2*3.5),0)
            except:
                # if no information is available, assume the height based on a average ground area to height ratio
                gdf_out.at[index, 'number_of_floors'] = round(math.sqrt(gdf_out.at[index,'projected_ground_area']) / (2*3.5),0)

    data_avail_number_of_floors = 1 - gdf_in['building:levels'].isna().sum() / len(gdf_in)
    cprint(f"Data availability for number of floors: {data_avail_number_of_floors*100:.2f} %")

    # calculate the number of dwellings
    for index, row in gdf_in.iterrows():
        try: 
            gdf_out.at[index, 'number_of_dwellings'] = int(row['building_flats']) 
        except:
            # if no information is available, assume the number of dwellings based on the size of the building
            gdf_out.at[index, 'number_of_dwellings'] = math.ceil(gdf_out.at[index, 'projected_ground_area'] * gdf_out.at[index, 'number_of_floors'] / 250) # assuming 250 m² per dwelling; including that there is a lot of projected area the is  not residnetial area
    
    data_avail_number_of_dwellings = 1 - gdf_in['building_flats'].isna().sum() / len(gdf_in)
    cprint(f"Data availability for number of dwellings: {data_avail_number_of_dwellings*100:.2f} %")

    # get the building category according to the OSM definition: https://wiki.openstreetmap.org/wiki/Key:building

    # split up multiple entries first
    for index, row in gdf_in.iterrows():
        multiple_entries = row['building'].split(';')
        if len(multiple_entries) == 0:
            gdf_out.at[index, 'building_primary'] = 'house' # default value as that is the most freuequently used building type
            gdf_out.at[index, 'building_secondary'] = 'None'
        elif len(multiple_entries) == 1:
            gdf_out.at[index, 'building_primary'] = multiple_entries[0]
            gdf_out.at[index, 'building_secondary'] = 'None'

        else:
            # check if one entry is a garage or carport and assign it to the secondary building type
            if 'carport' in multiple_entries[0] or 'garage' in multiple_entries[0] or 'garages' in multiple_entries[0]  or 'parking' in multiple_entries[0]: 
                gdf_out.at[index, 'building_secondary'] = 'garage'
                gdf_out.at[index, 'building_primary'] = multiple_entries[1]
            else:
                gdf_out.at[index, 'building_primary'] = multiple_entries[0]
                gdf_out.at[index, 'building_secondary'] = multiple_entries[1]


    #gdf_in['building_primary'] = gdf_in['building'].str.split(';', expand=True)[0] 
    #gdf_in['building_secondary'] = gdf_in['building'].str.split(';', expand=True)[1] 

    # iterate through the building types and assign a category
    ratio_heated_projected_area = 0.8 # ratio of heated area to projected area; this value is based on a rough estimation and can be adapted

    for index, row in gdf_out.iterrows():
        try:
            match row['building_primary']:
                case 'apartments':
                    gdf_out.at[index, 'building_category'] = 'AB'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'barracks':
                    gdf_out.at[index, 'building_category'] = 'TH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'bungalow': 
                    gdf_out.at[index, 'building_category'] = 'SFH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'cabin':
                    gdf_out.at[index, 'building_category'] = 'SFH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'detached':
                    gdf_out.at[index, 'building_category'] = 'SFH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'annexe':  
                    gdf_out.at[index, 'building_category'] = 'SFH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'dormitory':
                    gdf_out.at[index, 'building_category'] = 'AB'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'farm':
                    gdf_out.at[index, 'building_category'] = 'SFH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'ger':
                    gdf_out.at[index, 'building_category'] = 'other'
                    gdf_out.at[index, 'heated_area'] = 0
                case 'hotel':
                    gdf_out.at[index, 'building_category'] = 'AB' # assuming a hotel has a similar energy demand as an apartment building
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'house':
                    gdf_out.at[index, 'building_category'] = 'SFH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'houseboat':
                    gdf_out.at[index, 'building_category'] = 'SFH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'residential':
                    if gdf_out.at[index, 'number_of_dwellings'] < 2:
                        gdf_out.at[index, 'building_category'] = 'SFH'
                    elif gdf_out.at[index, 'number_of_dwellings'] < 5:
                        gdf_out.at[index, 'building_category'] = 'MFH'
                    else:
                        gdf_out.at[index, 'building_category'] = 'AB'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'semidetached_house':
                    gdf_out.at[index, 'building_category'] = 'TH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'static_caravan':
                    gdf_out.at[index, 'building_category'] = 'other'
                    gdf_out.at[index, 'heated_area'] = 0
                case 'stilt_house':
                    gdf_out.at[index, 'building_category'] = 'other'
                    gdf_out.at[index, 'heated_area'] = 0
                case 'terrace':
                    gdf_out.at[index, 'building_category'] = 'TH'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'tree_house':
                    gdf_out.at[index, 'building_category'] = 'other'
                    gdf_out.at[index, 'heated_area'] = 0
                case 'trullo':
                    gdf_out.at[index, 'building_category'] = 'other'
                    gdf_out.at[index, 'heated_area'] = 0
                case 'commercial':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'industrial':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'kiosk':
                    gdf_out.at[index, 'building_category'] = 'other'
                    gdf_out.at[index, 'heated_area'] = 0 
                case 'office':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'retail':  
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'supermarket':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'warehouse':
                    gdf_out.at[index, 'building_category'] = 'other'
                    gdf_out.at[index, 'heated_area'] = 0
                case 'civic':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'college':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'fire_station':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'government':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'hospital':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'kindergarten':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'museum':  
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'public':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'school':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'university':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'sports_hall':
                    gdf_out.at[index, 'building_category'] = 'PCI'
                    gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case 'yes': # this can cause problems, as the 'yes' tag is used very inconsistently in OSM, and therefore some buildings can be missmatched!
                    if gdf_out.at[index, 'number_of_dwellings'] < 1:
                        gdf_out.at[index, 'building_category'] = 'other'
                        gdf_out.at[index, 'heated_area'] = 0
                    elif gdf_out.at[index, 'number_of_dwellings'] < 5:
                        gdf_out.at[index, 'building_category'] = 'MFH'
                        gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                    else:
                        gdf_out.at[index, 'building_category'] = 'AB'
                        gdf_out.at[index, 'heated_area'] = gdf_out.at[index, 'projected_ground_area'] * ratio_heated_projected_area
                case default:
                    gdf_out.at[index, 'building_category'] = 'other'
                    gdf_out.at[index, 'heated_area'] = 0
        except:
            gdf_out.at[index, 'building_category'] = 'other'
            gdf_out.at[index, 'heated_area'] = 0
            print("BadDOG")
    

    # decrease heating area by parking space which is not heated
    gdf_out.loc[gdf_out['building_secondary'] != 'None','heated_area'] = gdf_out.loc[gdf_out['building_secondary'] != 'None', 'heated_area'] * 0.7
    
    cprint(f"Done: Data processing for building classification", "green")
    return gdf_out

def adjust_num_of_dwelling(gdf_in: gpd.GeoDataFrame):
    """"
    Adjust the number of dwellings for non residential buildings to zero
    param gdf_in: GeoDataFrame
    return: None
    """
    #remove the number of dwelling for non residential buildings
    gdf_in.loc[gdf_in['building_category'] == 'PCI', 'number_of_dwellings'] = 0
    gdf_in.loc[gdf_in['building_category'] == 'other', 'number_of_dwellings'] = 0


def merge_other_data(gdf_in: gpd.GeoDataFrame):
    """
    This function is a dummy to merge in other data sources like cadastre data if available.
    :param gdf_in: GeoDataFrame
    :return: GeoDataFrame containing the merged data
    :rtype: gpd.GeoDataFrame
    """
    gdf_out = gdf_in.copy()
    # add code here to merge in other data sources e.g. more accurate building data from the cadastre

    return gdf_out

    
def get_spec_HD(gdf_in: gpd.GeoDataFrame, path_spec_HD: str):
    """
    Get the specific heating demand of the buildings, based on the TABULA webtool: https://webtool.building-typology.eu/#bm
    The path_spec_HD is the path to the excel file containing the specific heating demand data. Those are based on the TABULA webtool and are Country specific. 
    To get the accurate data for a specific country, the data must be downloaded from the TABULA webtool and saved as an excel file in the provided format.
    :param gdf_in: GeoDataFrame containing the building data
    :param path_spec_HD: str to the file containing the specific heating demand data
    :return: None
    """
    # load data for specific heating demand
    spec_HD = pd.read_excel(path_spec_HD, index_col=[0,1])

    # iterate trough all rows and columns in the data
    for year in spec_HD.index:
        for bulding_type in spec_HD.columns:
        #print(row, column, data.loc[row, column])
            gdf_in.loc[(gdf_in['building_category'] == bulding_type) & (gdf_in['year_of_construction'] >= year[0]) & (gdf_in['year_of_construction'] <= year[1] ), 'specific_HD'] = spec_HD.at[year, bulding_type]
           
def get_yearly_HD(gdf_in: gpd.GeoDataFrame):
    """"
    Calculate the yearly heating demand based on the specific heating demand and the heated area
    param gdf_in: GeoDataFrame containing specific heating demand and heated area
    return: None
    """
    gdf_in['YearlyDemand'] = gdf_in['specific_HD'] * gdf_in['heated_area'] 


def estimate_thermal_properties(gdf_in: gpd.GeoDataFrame, heatinghours=50000, speHeatStorCap = 0.04, surface_to_area = 2.5):
    """
    Estimate the thermal properties of the buildings based on the specific heating demand and the heated area. The estimated thermal properties are scaled up for the whole building.
    :param gdf_in: GeoDataFrame
    :param heatinghours: int: number of heating hours per year. Sumed up the temperature difference between setpoint and outside temperature for all hours of the year.
    :param speHeatStorCap: float: specific heat storage capacity of the building material in kWh/m²K of a wall of average thickness
    :param surface_to_area: float ratio of thermal active area of the building (i.e. walls, roof, floor) to the ground area of the building (avoid double counting of indoor walls)
    :return None
    """
    gdf_in['GeneralisedThermCond'] = gdf_in['YearlyDemand'] / heatinghours
    gdf_in['GeneralisedThermCap'] = gdf_in['projected_ground_area'] * surface_to_area * speHeatStorCap
    gdf_in['MaxDemand'] = gdf_in['GeneralisedThermCond'] * 25 + gdf_in['GeneralisedThermCap'] * 2 #assuming the max heating power to heat at 25 °C temp. diff, and be able to rise temp by 2 °C per hour


def define_setpontTemp(gdf_in: gpd.GeoDataFrame, temp_setpoint = 22, std_setpoint = 2, temp_setback = 17, std_setback = 2):
    """
    Define the setpoint temperature and the setback temperature for the buildings. The values are randomly generated based on a normal distribution around the mean values.
    :param gdf_in: GeoDataFrame contrining the building data
    :param temp_setpoint: float: mean value for the setpoint temperature
    :param std_setpoint: float: standard deviation for the setpoint temperature
    :param temp_setback: float: mean value for the setback temperature
    :param std_setback: float: standard deviation for the setback temperature
    :return: None    
    """
    gdf_in['temp_setpoint'] =  random.normal(temp_setpoint, std_setpoint, len(gdf_in))
    gdf_in['temp_setback'] =  random.normal(temp_setback, std_setback, len(gdf_in))



def estimate_local_heat_prod_costs(gdf_in: gpd.GeoDataFrame, cost_per_kWh = 0.15):
    """
    Estimate the local heating production costs based on the cost per kWh. The costs are randomly generated based on a normal distribution around the mean value. If real data is available, this function can be replaced by the real data.
    :param gdf_in: GeoDataFrame containing the building data
    :param cost_per_kWh: float: mean value for the heating costs per kWh
    :return None
    """
    gdf_in['LocalHeatProdCosts'] =  random.normal(cost_per_kWh, 0.1, len(gdf_in))
        

def write_buildingdata_to_disk(gdf_in: gpd.GeoDataFrame, casestudy: str, config: dict):
    """
    Write the building data to disk as a GeoJSON file
    :param gdf_in: GeoDataFrame containing the building data
    :param casestudy: str: name of the casestudy
    :param config: dict: configuration dictionary
    :return: None
    """
    # check if directory exists
    data_path = os.path.join(casestudy)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    gdf_in.to_file(os.path.join(data_path,config['building_data']['buildings_gdf']), driver='GeoJSON')
    cprint(f"Done: Building data written to disk", "green")
    
   

if __name__ == "__main__":
    casestudy = "Frauental"
    location = "Frauental, Styria, Austria"
    

    # load config
    config = load_config()

    # generate a new case study folder if not exists
    generate_new_case_study(casestudy, config)
    
    # Get geodata for a place
    gdf_raw_geo = get_geodata_from_place(location)
    # Extract the relevant data
    gdf_relevant_data = extract_relevat_data(gdf_raw_geo)
    # Estimate the data
    gdf_estimated_data = estimate_data_osm(gdf_relevant_data)
    # Merge other data
    gdf_complete = merge_other_data(gdf_estimated_data)
    
    # Get the specific heating demand
    path_spec_HD = os.path.join(casestudy,config['parameter_dir'], config['building_data_dir'], config['case_study_data']['building_typology'])

    get_spec_HD(gdf_complete, path_spec_HD)
    # Get the yearly heating demand
    get_yearly_HD(gdf_complete)
    # Estimate the thermal properties
    estimate_thermal_properties(gdf_complete)
    # Define the setpoint temperature
    define_setpontTemp(gdf_complete)

    # Estimate the local heating production costs   
    estimate_local_heat_prod_costs(gdf_complete)
    # Write the building data to disk
    write_buildingdata_to_disk(gdf_complete, casestudy, config)
