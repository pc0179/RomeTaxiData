# I really can't wait for the day where I can just use gits to do my work
# GLOBAL variables: datum position, start of simulation.... 
# note, nrows = sum(1 for _ in open('/home/pdawg/Old/Rome/all_rome_taxi_february.txt'))
# nrows = 21817851 #(3*11*17*38891)
# http://pandas-docs.github.io/pandas-docs-travis/io.html#iterating-through-files-chunk-by-chunk



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

#Setup Processed DataFrame with columns etc...
#new_dfcols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','LatLong','xy_pos']

new_dfcols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','Lat','Long','x','y']
new_dftrace = pd.DataFrame(columns=new_dfcols)
new_dftrace.to_csv('test_trace.csv', header=new_dfcols, index = True, sep=";")

#Obtain Chunk of Data from text file
raw_trace_data_filename = '/home/pdawg/RomeTaxiData/medium_raw_taxi_trace.csv' # all_rome_taxi_february.txt' #'trace100.txt'
reader=pd.read_table(raw_trace_data_filename,sep=";",chunksize=500 ,header = None, iterator=True)

for chunk in reader:
	
	#new_dfcols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','LatLong','xy_pos']
	#new_dftrace = pd.DataFrame(columns=new_dfcols)
	new_dftrace = pd.DataFrame()
	
	dftrace = chunk # reader.get_chunk()
	dftrace.columns = ['taxi_id','ts','gps']
	
		#TaxID
	new_dftrace['taxi_id'] = dftrace['taxi_id']

	#TimeStamp post-processing
	new_dftrace['ts_dt'] = dftrace['ts'].apply(tsconv.PyTimeConv)
	new_dftrace['sim_t'] = new_dftrace['ts_dt'].apply(tsconv.SimTimeSeconds)
	new_dftrace['sim_day_num'] = new_dftrace['ts_dt'].apply(tsconv.SimDayNum)
	new_dftrace['weekday_num'] = new_dftrace['ts_dt'].apply(tsconv.SimWeekDayNum)

	#GPS/Position Post-processing
	#new_dftrace['LatLong'] = dftrace['gps'].apply(xyconv.LatLongConv)
	#new_dftrace['xy_pos'] = new_dftrace['LatLong'].apply(xyconv.Position_From_Datum) 
	new_dftrace['Lat'] = dftrace['gps'].apply(xyconv.LatConv2)
	new_dftrace['Long'] = dftrace['gps'].apply(xyconv.LongConv2)
	new_dftrace['x'] = new_dftrace.apply(lambda x: xyconv.XPos_From_Datum2( x['Lat'],x['Long']), axis=1, raw=True)
	new_dftrace['y'] = new_dftrace.apply(lambda x: xyconv.YPos_From_Datum2( x['Lat'], x['Long']), axis=1, raw=True)
	

# SO CLOSE!!! need to maybe have an 'initial csv file, with headers etc... then keep appending..., avoid having headers every chunk!
	new_dftrace.to_csv('test_trace.csv', mode='a', index = True, sep=";",header=False)
	





#import dask.dataframe as dd
#ddf = dd.read_csv('/home/pdawg/Old/Rome/all_rome_taxi_february.txt')
#ddf.columns = ['taxi_id','ts','gps']
#daskdf = dd.read_csv('/home/pdawg/RomeTaxiData/mini_trace.txt')
#dftrace = daskdf



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
