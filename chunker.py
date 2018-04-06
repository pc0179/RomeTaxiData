# I really can't wait for the day where I can just use gits to do my work
# GLOBAL variables: datum position, start of simulation.... 


import pandas as pd
import timestamp_conv  as tsconv
import xy_coordinate_conv as xyconv

#Global Variables
import RomeTaxiGlobalVars

#global DatumLong
#global DatumLat
#global sim_start_time

#sim_start_time = '2014-02-01 00:00:00'
#DatumLong = 12.492373
#DatumLat = 41.890251

#Obtain Chunk of Data from text file
reader=pd.read_table('mini_trace.txt',sep=";",chunksize=5, header = None)

dftrace = reader.get_chunk()
dftrace.columns = ['taxi_id','ts','gps']

#Setup Processed DataFrame with columns etc...
new_dfcols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','LatLong','xy_pos']
new_dftrace = pd.DataFrame(columns=new_dfcols)

#TaxID
new_dftrace['taxi_id'] = dftrace['taxi_id']

#TimeStamp post-processing
new_dftrace['ts_dt'] = dftrace['ts'].apply(tsconv.PyTimeConv)
new_dftrace['sim_t'] = new_dftrace['ts_dt'].apply(tsconv.SimTimeSeconds)
new_dftrace['sim_day_num'] = new_dftrace['ts_dt'].apply(tsconv.SimDayNum)
new_dftrace['weekday_num'] = new_dftrace['ts_dt'].apply(tsconv.SimWeekDayNum)

#GPS/Position Post-processing
new_dftrace['LatLong'] = dftrace['gps'].apply(xyconv.LatLongConv)
new_dftrace['xy_pos'] = new_dftrace['LatLong'].apply(xyconv.Position_From_Datum)



# TimeStamp Pre-processing

#b = dftrace['ts'].apply(tsconv.PyTimeConv)


#cool line... should aim to make changes to entire chunk, then  save... 
#dftrace['gps']=dftrace['gps'].apply(xyconv.LatLongConv)


#list_id = list(dftrace[0])
#list_ts = list(dftrace[1])
#list_gps = list(dftrace[2])

#import PyTimeConv as tsconv
#b = []
#for i in range(len(list_ts)):
#	b.append(tsconv.PyTimeConv(list_ts[i]))

#Okay, doing well, sofar.... 
#Need to make the loop above faster/more efficient.
#test with GPS points...
# then need to get chunking and processing properly
# Friday: working SQL/postgis database
# weekend? - plotting graphs, cool queries? and some writing for ParkUs...
# Monday - animation, more graphs... 
# Tuesday - likely more graphs n processing, btw what does SFdata look like?
# Wednesday tidy + meeting...







#for chunk in reader:
#	chunk.columns = ['taxi_id','ts','gps_pos']




#chunks=pd.read_csv('mini_trace.txt',chunksize=2,sep=';', names=['taxi_id','ts','gps_pos'], header = None)

#reader=pd.read_table('mini_trace.txt',sep=";",chunksize=5,columns = ['taxi_id','ts','gps_pos'], header = None)
#df=pd.DataFrame()
#print(df)
#time df=pd.concat(chunk.groupby(['lat','long',chunk['date'].map(lambda x: x.year)])['rf'].agg(['sum']) for chunk in chunks)
