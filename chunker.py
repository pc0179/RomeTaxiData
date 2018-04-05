import pandas as pd

reader=pd.read_table('mini_trace.txt',sep=";",chunksize=5, header = None)
dftrace = reader.get_chunk()

list_id = list(dftrace[0])
list_ts = list(dftrace[1])
list_gps = list(dftrace[2])

import PyTimeConv as tsconv
b = []
for i in range(len(list_ts)):
	b.append(tsconv.PyTimeConv(list_ts[i]))

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
