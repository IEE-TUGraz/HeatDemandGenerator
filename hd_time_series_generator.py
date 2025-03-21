import numpy as np
import pandas as pd
from termcolor import cprint
import yaml
import os

def generate_mcchain(transion_matrix: pd.DataFrame, ini_prob = 0.2, resample = True):
    """
    generate a markov chain with a given transition matrix and a given initial probability
    :param transion_matrix: pandas dataframe with the transition probabilities
    :param ini_prob: initial probability for the first state
    :param resample: if True the data is resampled to hourly data
    :return: pandas dataframe with the markov chain
    :rtype: pandas.DataFrame
    """

    mchain = pd.DataFrame(columns=['timestep','state'], index=range(len(transion_matrix)))

    # define a random starting value with a given probability
    if np.random.rand() < ini_prob:
        ini_state = True
    else:
        ini_state = False

    mchain.loc[0] = [0,ini_state]

    # generate the markov chain
    for idx, row in mchain.iterrows():
        if idx == 0:
            continue
        else:
            r = np.random.random()
            if mchain.loc[idx-1,'state'] == False:
                if r < transion_matrix.loc[idx-1,'0to1']: 
                    mchain.loc[idx, 'state'] = True
                else:
                    mchain.loc[idx, 'state'] = False
            elif mchain.loc[idx-1,'state'] == 1:
                if r < transion_matrix.loc[idx-1,'1to0']:
                    mchain.loc[idx, 'state'] = False
                else:
                    mchain.loc[idx, 'state'] = True
        row['timestep'] = idx

    # resample it to hourly data
    if resample:
        mchain_res = (mchain.groupby(mchain.index // 6).mean())
        mchain_res['timestep'] = mchain_res.index

        # round the values 
        mchain_res['state'] = mchain_res['state'].apply(lambda x: bool(round(x, 0)))
        return mchain_res
    else:
        return mchain

def prepare_temp_out(scenario: str, config: dict):
    """ 
    Prepare the outside temperature data for the simulation
    :param scenario: path to the scenario folder
    :param config: configuration dictionary
    :return: pandas dataframe with the outside temperature
    :rtype: pandas.DataFrame
    """
    path_temp = os.path.join(scenario, config["parameter_dir"], config["MCMC_dir"], config['case_study_data']['outside_temp'])

    df_temperature = pd.read_excel(path_temp) #, skiprows=[0,1,2])
    df_temperature['time'] = pd.to_datetime(df_temperature['time'])

    # add info if it is a buisness day or not
    df_temperature['businesday'] = df_temperature['time'].apply(lambda x: x.weekday() <= 4)

    # rename the columns
    df_temperature.columns = ['time', 'temperature', 'businesday']

    return df_temperature

def calculate_activ_occup(df_temperature: pd.DataFrame, df_tansition_matrix_WD: pd.DataFrame, df_transition_matrix_WE: pd.DataFrame, ini_prob = 0.2):
    """
    Calculate the active occupancy for each time step based on the transition matrix and MCMC provess
    :param df_temperature: pandas dataframe with the temperature data
    :param df_tansition_matrix_WD: pandas dataframe with the transition matrix for the weekdays
    :param df_transition_matrix_WE: pandas dataframe with the transition matrix for the weekends
    :param ini_prob: initial probability for the first state
    :return: pandas dataframe with the active occupancy
    :rtype: pandas.DataFrame
    """
    
    # calculate the markov chain for the weekdays and weekends once for one day
    mchainWD = generate_mcchain(df_tansition_matrix_WD, ini_prob)
    mchainWE = generate_mcchain(df_transition_matrix_WE, ini_prob)

    # merge the days over the whole year considering business days and weekends
    for day in range(0, int(len(df_temperature)/len(mchainWD))):
        if df_temperature.loc[day*24,'businesday'] == True:
            df_temperature.loc[day*24:day*24+23,'active_occ'] = (mchainWD['state'].values)
        else:
            df_temperature.loc[day*24:day*24+23,'active_occ'] = (mchainWE['state'].values)
    return df_temperature
    
def calculate_setpoint_temp(df_temperature: pd.DataFrame, T_setpoint = 22, T_setback = 18):
    """
    Calculate the setpoint temperature based on the active occupancy and the setpoint and setback temperature.
    :param df_temperature: pandas dataframe with the temperature data
    :param T_setpoint: setpoint temperature
    :param T_setback: setback temperature
    :return: pandas dataframe with the setpoint temperature
    :rtype: pandas.DataFrame 
    """
    df_temperature['setpoint'] = df_temperature['active_occ'].apply(lambda x: T_setpoint if x == True else T_setback)
    return df_temperature


# calculating all timesteps for every dwelling is very time consuming. 
# Might be improved by to records methode or numba or reformulating the function

def calculate_hd_ts(df_temperature: pd.DataFrame, max_heat_power=15, thermal_coductance=0.12, thermal_storage_capacity=7, internal_gain=0):
    """   
    Calculate the heating demand for each time step based on the temperature data and the building parameters"
    :param df_temperature: pandas dataframe with the temperature data
    :param max_heat_power: maximum heating power
    :param thermal_coductance: thermal coductance
    :param thermal_storage_capacity: thermal storage capacity
    :param internal_gain: internal gain
    :return: pandas dataframe with the heating demand
    :rtype: pandas.DataFrame
    """

    df_heating = pd.DataFrame(columns=['time', 'T_out', 'T_set', 'internal_gain', 'T_in', 'heat_demand', 'actual_heat_power'], index=range(len(df_temperature)))
    df_heating['time'] = df_temperature['time']
    df_heating['T_out'] = df_temperature['temperature']
    df_heating['T_set'] = df_temperature['setpoint']
    df_heating['internal_gain'] = internal_gain * df_temperature['active_occ']

    # set the values for the time step t-1
    T_in_prev = df_heating.loc[0, 'T_set'] + (np.random.randn() * 0.1)
    T_out_prev = df_heating.loc[0, 'T_out']
    actual_heating_power_prev = 0

    for row in df_heating.itertuples():
        Q_losses = thermal_coductance*(T_in_prev - T_out_prev) 
        T_in = T_in_prev + (actual_heating_power_prev + row.internal_gain - Q_losses) / thermal_storage_capacity
        
        #df_heating.loc[idx,'T_in'] = T_in

        heating_for_T_change = thermal_storage_capacity * (row.T_set - T_in) # required heating to change the temperature
        
        heat_demand = thermal_coductance * (T_in- row.T_out) + heating_for_T_change

        actual_heating_power = max(0, min(max_heat_power, heat_demand)) 

        # set the calculated vales as new values for the next iteration
        T_in_prev = T_in
        T_out_prev = row.T_out
        actual_heating_power_prev = actual_heating_power
      
        idx = row.Index
        df_heating.at[idx, 'T_in'] = T_in
        df_heating.at[idx, 'heat_demand'] = heat_demand
        df_heating.at[idx, 'actual_heat_power'] = actual_heating_power

    return df_heating


def calculate_avg_hd_ts(df_temperature, T_setpoint, T_setback, df_tansition_matrix_WD, df_transition_matrix_WE, num_dwellings=1, max_heat_power=10, thermal_coductance=0.12, thermal_storage_capacity=7, internal_gain=0):
    """ 
    Calculate the average heating demand for a given number of dwellings
    :param df_temperature: pandas dataframe with the temperature data
    :param T_setpoint: setpoint temperature
    :param T_setback: setback temperature
    :param df_tansition_matrix_WD: pandas dataframe with the transition matrix for the weekdays
    :param df_transition_matrix_WE: pandas dataframe with the transition matrix for the weekends
    :param num_dwellings: number of dwellings
    :param max_heat_power: maximum heating power
    :param thermal_coductance: thermal coductance
    :param thermal_storage_capacity: thermal storage capacity
    :param internal_gain: internal gain
    :return: pandas dataframe with the heating demand
    :rtype: pandas.DataFrame
    """
    
    df_list = []

    # set a upper limit for the number of dwellings and reduce every value to this maximum
    upper_lim_num_of_dwellings = 10 # a higher number of dwellings increases the computational effort, but does not add much for averaging out
    num_dwellings = min(num_dwellings, upper_lim_num_of_dwellings)


    if num_dwellings >= 1:
        for runs in range(int(num_dwellings)):
            df_temperature =  calculate_activ_occup(df_temperature, df_tansition_matrix_WD, df_transition_matrix_WE, ini_prob = 0.2)
            df_temperature =  calculate_setpoint_temp(df_temperature, T_setpoint, T_setback)
        
            df_list.append(calculate_hd_ts(df_temperature, max_heat_power/num_dwellings, thermal_coductance/num_dwellings, thermal_storage_capacity/num_dwellings, internal_gain/num_dwellings))

        # calculate the average of all dataframes
        df_heating = df_list[0]
        for df in df_list[1:]:
            df_heating['T_set'] = df_heating['T_set'] + df['T_set']
            df_heating['internal_gain'] = df_heating['internal_gain'] + df['internal_gain']
            df_heating['T_in'] = df_heating['T_in'] + df['T_in']
            df_heating['heat_demand'] = df_heating['heat_demand'] + df['heat_demand']
            df_heating['actual_heat_power'] = df_heating['actual_heat_power'] + df['actual_heat_power']
        
        df_heating['T_set'] = df_heating['T_set']/num_dwellings
        df_heating['T_in'] = df_heating['T_in']/num_dwellings
            
        return df_heating
    
    else:
        #tbd create a fixed df_temperature for industry and public buildings
        df_heating =  calculate_hd_ts(df_temperature, max_heat_power, thermal_coductance, thermal_storage_capacity, internal_gain)

        return df_heating
    
    
def generate_building_time_series(gdf_buildings, df_temperature, df_transition_matrix_WD, df_transition_matrix_WE):
    """
    Generate the heating demand time series for all buildings for a given dataframe with the building parameters
    :param gdf_buildings: geopandas dataframe with the building parameters
    :param df_temperature: pandas dataframe with the temperature data
    :param df_transition_matrix_WD: pandas dataframe with the transition matrix for the weekdays
    :param df_transition_matrix_WE: pandas dataframe with the transition matrix for the weekends
    :return: pandas dataframe with the heating demand time series
    :rtype: pandas.DataFrame
    """
    
    # generate a data frame for all buildings with the building ids as columns
    col_names = np.append('hour', gdf_buildings['building_id'].values)
    df_buidling_TS = pd.DataFrame(columns=col_names)
    df_buidling_TS['hour'] = range(1, len(df_temperature) + 1)
    
    cprint(F"Start generating {int(gdf_buildings['number_of_dwellings'].sum())} time series for {len(gdf_buildings)} buildings")

    for row in gdf_buildings.itertuples():
        if row.YearlyDemand != 0:
            # calculate the heating demand for each building
            df_buidling_TS[row.building_id] = calculate_avg_hd_ts(
                df_temperature, row.temp_setpoint, row.temp_setback, 
                df_transition_matrix_WD, df_transition_matrix_WE, 
                row.number_of_dwellings, row.MaxDemand, 
                row.GeneralisedThermCond, row.GeneralisedThermCap, 0
            )['actual_heat_power'].values
        else:
            df_buidling_TS[row.building_id] = 0

        if row.Index % 50 == 0:
            print(f"In progress: Building {row.Index} of {len(gdf_buildings)} calculated")

        '''  
        # for debugging
        print("yearly demand", row.YearlyDemand)
        print("yearly sum calculated", df_buidling_TS[row.building_id].sum())

        if row.Index == 5:
            break'''
    
    return df_buidling_TS


def write_hdts_to_disk(df_heating, casestudy: str, config: dict):
    """"
    Write the heating demand time series to disk"
    :param df_heating: pandas dataframe with the heating demand"
    :param casestudy: path to the casestudy folder""
    :param config: configuration dictionary
    :return: None
    """

    # check if directory exists
    data_path = os.path.join(casestudy)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    df_heating.to_csv(os.path.join(data_path,config['building_data']['building_TS']), index=False, sep=',')



if __name__ == "__main__":
    
    pass
            