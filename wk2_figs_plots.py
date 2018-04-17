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

connection = psycopg2.connect(connect_str)

#taxidf = pdsql.read_sql_query(execution_str,connection)

#--------------------------------------------

# Now for numero of taxis per day in sim...

"""

sim_day_num = list(range(27)) # for there are 28 days of taxi trace data
connection = psycopg2.connect(connect_str)
taxiIDs_onduty = []
num_taxis_per_day = []
for day_number in sim_day_num:

	execution_str = ("SELECT DISTINCT taxi_id FROM rometaxidata WHERE sim_day_num = %s" % (str(day_number)))
	uniq_taxidf = pdsql.read_sql_query(execution_str,connection)
	taxiIDs_onduty.append(list(uniq_taxidf)) #just for jks, would be nice to have a list of taxi_ID numbers to search for on a given day
	num_taxis_per_day.append(len(uniq_taxidf))
	print(day_number)

df_num_taxis_per_day = pd.DataFrame(num_taxis_per_day)
df_num_taxis_per_day.to_csv('/home/pdawg/RomeTaxiData/number_taxis_per_day.csv')

df_taxis_on_duty = pd.DataFrame(taxiIDs_onduty)
df_taxis_on_duty.to_csv('/home/pdawg/RomeTaxiData/taxiIDs_on_duty.csv')

plt.plot(df_num_taxis_per_day,'o-')
plt.xlabel('Taxi Trace Day Number')
plt.ylabel('Total number of taxis on duty')
plt.show()

"""

# now number of taxis per hour...
# SELECT COUNT(*) FROM (SELECT DISTINCT taxi_id FROM rometaxidata WHERE (sim_t BETWEEN 0 AND 3600)) AS foo;
sim_times = list(range(0,60*60*24*27,60*60)) #- everyhour of sim, 60*60s
num_taxis_per_hour = []
hour_counter = []

for sim_t in sim_times:
	#execution_str = ("SELECT COUNT(*) FROM (SELECT DISTINCT taxi_id FROM rometaxidata WHERE (sim_t BETWEEN %s AND %s)) AS foo" % (str(sim_t),str(sim_t+60*60)))
	#execution_str = ("SELECT taxi_id, COUNT(*) AS number_taxis_on_duty FROM rometaxidata WHERE (sim_t BETWEEN %s AND %s) GROUP BY taxi_id" % (str(sim_t),str(sim_t+60*60)))
	execution_str = ("SELECT COUNT(DISTINCT taxi_id) FROM rometaxidata WHERE (sim_t BETWEEN %s AND %s)" % (str(sim_t),str(sim_t+60*60)))
	count_taxidf = pdsql.read_sql_query(execution_str,connection)
	num_taxis_per_hour.append(int(count_taxidf['count']))
	hour_counter.append(sim_t)
	print(round((sim_t/sim_times[-1])*100))	


np.savetxt("/home/pdawg/RomeTaxiData/taxi_count_per_hour.csv", num_taxis_per_hour, delimiter=",")

df_taxis_hour = pd.DataFrame({"sim_t": hour_counter, "num_taxis": num_taxis_per_hour})
df_taxis_hour.to_csv("/home/pdawg/RomeTaxiData/pd_taxi_count_per_hour.csv", index=False)




""" SECTION for creating update frequency histogram/pdf/CDF

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

"""





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
