"""
slightly more efficient 'chunker.py' script
pc0179 on the Kboard
python 3+ throughout

1. load basic/raw csv file
2. filter duplicates
3. add intelligent columns (later for database)
4. output - csv or straight to db?

5. once in db...
- drop rows outside of bbox?
- take average of rows with identical time_stamps...
- some estimate of (instant?) velocity

"""

import pandas as pd
import datetime as DT
import numpy as np
import RomeTaxiGlobalVars as RTGV
import re


#--------------------------
# Temporal funcs...

global sim_start_time
sim_start_time = DT.datetime.strptime(RTGV.sim_start_time, '%Y-%m-%d %H:%M:%S')


def PyDateTimeConv(csv_time):
	chopped_csv_time = csv_time[:-3]
	if len(chopped_csv_time)<20:
		py_time = DT.datetime.strptime(chopped_csv_time,'%Y-%m-%d %H:%M:%S')
	else:
		py_time_ms = DT.datetime.strptime(chopped_csv_time,'%Y-%m-%d %H:%M:%S.%f')
		py_time = RoundTimeSeconds(py_time_ms)
	return py_time

def RoundTimeSeconds(some_DT_obj):
	if some_DT_obj.microsecond>= 5e5:
		some_DT_obj = some_DT_obj + DT.timedelta(seconds=1)
	return some_DT_obj.replace(microsecond=0)

def SimuTime(some_DT_obj):
#	sim_start_time = '2014-02-01 00:00:00'
	#if sim_start_time == None:
	#	sim_start_time =
	global sim_start_time
	sim_time_s = (some_DT_obj-sim_start_time).total_seconds()
	sim_daynum = (some_DT_obj-sim_start_time).days()
	sim_weekday = some_DT_obj.weekday()
	return sim_time_s,sim_daynum,sim_weekday

def SimDayNum(some_DT_obj):
	global sim_start_time
	#sim_daynum = (some_DT_obj-sim_start_time).days()
	sim_daynum = some_DT_obj.day - sim_start_time.day
	return int(sim_daynum)

def SimWeekDayNum(some_DT_obj):
	sim_weekday = some_DT_obj.weekday()
	return sim_weekday

def SimTimeSeconds(some_DT_obj):
	global sim_start_time
	sim_time_s = (some_DT_obj-sim_start_time).total_seconds()
	return int(sim_time_s)

def PyUnixTimeConv(some_DT_obj):
    """ func to convert DateTime Obj to Unix Epoch time, in SECONDS, hence rounding"""
    output_unix_time = some_DT_obj.timestamp()
    return round(output_unix_time)



# ---------------------------------------
# spatial stuff/funcs


def LatConv2(GPS_str):
	numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
	rx = re.compile(numeric_const_pattern, re.VERBOSE)
	a = rx.findall(GPS_str)
	lat1 = round(float(a[0]),6)
	return lat1

def LongConv2(GPS_str):
	numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
	rx = re.compile(numeric_const_pattern, re.VERBOSE)
	a = rx.findall(GPS_str)
	long1 = round(float(a[1]),6)
	return long1


def haversine_pc(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    Hdistance = 6371e3*c  #working in metres!
    return Hdistance


def LatLong2XYConv(latitude,longitude,datumlat,datumlong):
    
    y = round(spherical_dist([latitude,datumlong],[datumlat,datumlong]))
    x = round(spherical_dist([datumlat,longitude],[datumlat,datumlong]))
    
    if latitude<datumlat: #IMPLIES  taxi is south of datum (South=-ve),(North=+ve)
        y = -y
    if longitude<datumlong: #IMPLIES taxi is west of datum (West=-ve) ,(East=+ve)
        x = -x

    return [x, y]

def Long2XConv(longitude):
    datumlat = RTGV.DatumLat
    datumlong = RTGV.DatumLong
    x = round(haversine_pc(datumlat,longitude,datumlat,datumlong))
    if longitude<datumlong: #IMPLIES taxi is west of datum (West=-ve) ,(East=+ve)
        x = -x
    return x

def Lat2YConv(latitude):
    datumlat = RTGV.DatumLat
    datumlong = RTGV.DatumLong
    y = round(haversine_pc(latitude,datumlong,datumlat,datumlong))
    if latitude<datumlat: #IMPLIES  taxi is south of datum (South=-ve),(North=+ve)
        y = -y
    return y

    

new_dfcols = ['taxi_id','dt_ts','unix_ts','weekday','trace_day','latitude','longitude','x','y']
new_tracedf = pd.DataFrame(columns=new_dfcols)

#outputfile
output_trace_file = '/home/user/RomeTaxiData/initial_filtered_rome_trace.csv'
new_tracedf.to_csv(output_trace_file, header=new_dfcols, index = False, sep=";")

#Obtain Chunk of Data from text file
raw_trace_data_filename = '/home/user/Downloads/c207_all_taxi_datasets/rome_taxi_trace_feb.txt'


reader=pd.read_table(raw_trace_data_filename,sep=";",chunksize=1000 ,header = None, iterator=True)
chunk_index = 0
for chunk in reader:
    chunk_index +=1
    new_tracedf = pd.DataFrame()
    new_tracedf = pd.DataFrame(columns=new_dfcols)
    tracedf = chunk

# duplicate removal, still haven't decided what to do with entries with identical ts but different long/lats...
#remove entries outside of bounding box later?

    tracedf = tracedf.drop_duplicates()


    tracedf.columns = ['taxi_id','ts','gps']
# taxi ID's    
    new_tracedf['taxi_id'] = tracedf['taxi_id']

# Trace data timestamps
    new_tracedf['dt_ts'] = tracedf['ts'].apply(PyDateTimeConv)
    new_tracedf['unix_ts'] = new_tracedf['dt_ts'].apply(PyUnixTimeConv)
    new_tracedf['weekday'] = new_tracedf['dt_ts'].apply(SimWeekDayNum)
    new_tracedf['trace_day'] = new_tracedf['dt_ts'].apply(SimDayNum)

# Taxi locations
    new_tracedf['latitude'] = tracedf['gps'].apply(LatConv2)
    new_tracedf['longitude'] = tracedf['gps'].apply(LongConv2)
    
    new_tracedf['x'] = new_tracedf['longitude'].apply(Long2XConv)
    new_tracedf['y'] = new_tracedf['latitude'].apply(Lat2YConv)


    new_tracedf.to_csv(output_trace_file, mode='a', index = False, sep=";",header=False)


