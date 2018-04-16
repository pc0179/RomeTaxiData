import psycopg2
#import csv
import pandas.io.sql as pdsql
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


#taxi_id list,
taxi_ids = pd.read_csv('/home/pdawg/RomeTaxiData/all_rome_taxi_ids.csv', header=None, sep="\n")
list_taxi_ids = list(taxi_ids[0]) #I was bored and this seemed easier than figuring out exactly how pandas.iterrows() bullshit works

connect_str = "dbname='taxitraces' user='postgres' host='localhost' password='postgres'"
# quick reminder of available columns in database:
# cols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','Lat1','Long1','x','y']

#interesting queries...
#"SELECT * FROM rometaxidata WHERE taxi_id =225"
#execution_str = "SELECT sim_t, Lat1, Long1 FROM rometaxidata WHERE taxi_id = 225"

#execution_str = "SELECT DISTINCT taxi_id FROM rometaxidata"

#execution_str = "SELECT sim_t, x, y FROM rometaxidata WHERE taxi_id = %s"

#"SELECT * FROM rometaxidata WHERE sim_day_num=1"
#execution_str = "SELECT * FROM rometaxidata WHERE (x BETWEEN -1000 AND 1000) AND (y BETWEEN -1000 AND 1000)"
#execution_str = "SELECT * FROM rometaxidata WHERE sim_day_num = 10 AND (x BETWEEN -1000 AND 1000) AND (y BETWEEN -1000 AND 1000)" 
#execution_str = "SELECT * FROM rometaxidata WHERE weekday_num = 0 AND taxi_id = 225"
#execution_str = "SELECT DISTINCT taxi_id FROM rometaxidata"

#connection = psycopg2.connect(connect_str)

#taxidf = pdsql.read_sql_query(execution_str,connection)


max_dt = int(4*60*60) #i.e. 4 hours, 14400 seconds, if time between points is greater, then ignore?
# bin widths in seconds
HBins = [0,2,5,6,7,8,9,10,12,15,20,30,60,120,300,600,1200,1800,3600,7200,max_dt]
overall_update_freq = np.zeros([1,len(HBins)-1])
connection = psycopg2.connect(connect_str)


for taxi_id in list_taxi_ids:

	execution_str = ("SELECT sim_t FROM rometaxidata WHERE taxi_id = %s" % (str(taxi_id)))

	taxidf = pdsql.read_sql_query(execution_str,connection)

	# for difference in update frequency:
	t_diff = taxidf['sim_t'].diff() #btw, I ckecked and it seems always monotonic, ie no need for sorting
	#t_diff_counts = (t_diff[t_diff<max_dt]).value_counts().sort_index()
	#t_diff_freq = t_diff_counts/t_diff_counts.sum()

	t_diff_list = list(t_diff[1:]) #removes NaN at the start of diff dataframe, converst to list/numpy array for histogram processing
	update_freq, more_bins = np.histogram(t_diff_list,bins=HBins)

	overall_update_freq += update_freq

	print(taxi_id)

df_update_freq = pd.DataFrame(overall_update_freq)
df_update_freq.to_csv("/home/pdawg/RomeTaxiData/overall_taxi_update_freq.csv", header=HBins[1:])

# histogram/frequency plot
x_mids = (np.array(HBins[1:]) + np.array(HBins[:-1]))/2
y_freq = np.transpose(overall_update_freq/np.sum(overall_update_freq))

plt.plot(x_mids,y_freq,'*-')
plt.xlabel('Diff. between taxi position updates/[seconds]')
plt.ylabel('Frequency')
plt.show()

y_cum = np.cumsum(overall_update_freq)/np.sum(overall_update_freq)

plt.plot(x_mids,np.transpose(y_cum)*100,'o-')
plt.xlabel('Diff. between taxi position updates/[seconds]')
plt.ylabel('Cumulative, percentage of trace data')
plt.show()



#fig, ax = plt.subplots()
#t_diff_freq.plot(ax=ax, kind='bar')
#plt.show()
# add to OVERALL frequency update histogram:

#np.savetxt("/home/pdawg/RomeTaxiData/overall_taxi_update_freq.csv", overall_update_freq, delimiter=",")

#df_update_freq = pd.DataFrame(np.transpose(overall_update_freq))
#df_update_freq.to_csv("/home/pdawg/RomeTaxiData/overall_taxi_update_freq.csv", header=HBins)

#taxidf.to_csv('all_rome_taxi_ids2.csv')

#frequency plotting within pandas???? crazy...
#fig, ax = plt.subplots()
#data['Points'].value_counts().plot(ax=ax, kind='bar')




# section below for map plotting
"""
#fig, ax = plt.subplots(figsize=(10,15))

fig = plt.gcf()

m = Basemap(
	resolution = 'c',
	projection = 'merc',
	llcrnrlon=12.442, llcrnrlat= 41.856, urcrnrlon=12.5387, urcrnrlat= 41.928)

#(12.442 41.856, 12.5387 41.928, 12.442 41.928, 12.5387 41.856, 12.442 41.856)

#m.fillcontinents(color='#f0f0f0')

road_shp_file = '/home/pdawg/RomeOSMdata/extract/extracted_Roman_roads'

m.readshapefile(road_shp_file, 'roads', drawbounds = True, color='grey')
x, y = m(list(taxidf.long1),list(taxidf.lat1))

plt.scatter(x,y, 5, marker='*', color='red')

#m.plot(taxidf.long1,taxidf.lat1,'b*')

plt.show()

#plt.savefig('/home/pdawg/RomeTaxiData/rome_road_network.png')
"""
