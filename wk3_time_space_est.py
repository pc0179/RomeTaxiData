"""
# week3 madness...
# filter, process (map-match) and output data to new postgres database
# aim by thurs, to be able to answer question, where are (best-guess/estimate) all the taxis at time T.
# then, work out, how far each one is from one another, likely a) search with in BBox, then b) do some fast osrm routing (get that line of sight distance?)


# to aid map-matching, best to use traces of one taxi at at time (allows for timestamp to be used... better approx. should result.)

# get a list of all taxi_IDs... within dataset
#execution_str = "SELECT DISTINCT taxi_id FROM rometaxidata"

#currently all designed to run on C207...

"""


import psycopg2
import pandas.io.sql as pdsql
import pandas as pd
from sqlalchemy import create_engine
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import numpy as np
import osrm


#1. querying database

# connection string for working on c207: connect_str = "dbname='rometaxitraces' user='postgres' host='localhost' password='postgres'"

# connection string for Klara:



# taxi_ids = pd.read_csv('/home/user/RomeTaxiData/all_rome_taxi_ids.csv', header=None, sep="\n")
# list_taxi_ids = list(taxi_ids[0]) #I was bored and this seemed easier than figuring out exactly how pandas.iterrows() bullshit works

#osrm.RequestConfig.host = "http://localhost:5000"

# quick reminder of available columns in database:
# cols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','Lat1','Long1','x','y','unix_ts']

#interesting queries...
#"SELECT * FROM rometaxidata WHERE taxi_id =225"
#execution_str = "SELECT sim_t, Lat1, Long1 FROM rometaxidata WHERE taxi_id = 225"

#execution_str = "SELECT DISTINCT taxi_id FROM rometaxidata"

#execution_str = "SELECT sim_t, x, y FROM rometaxidata WHERE taxi_id = %s"

#"SELECT * FROM rometaxidata WHERE sim_day_num=1"
#execution_str = "SELECT * FROM rometaxidata WHERE (x BETWEEN -1000 AND 1000) AND (y BETWEEN -1000 AND 1000)"
#execution_str = "SELECT * FROM rometaxidata WHERE sim_day_num = 10 AND (x BETWEEN -1000 AND 1000) AND (y BETWEEN -1000 AND 1000)" 
#execution_str = "SELECT * FROM rometaxidata WHERE weekday_num = 0 AND taxi_id = 225"


#execution_str = ("SELECT DISTINCT taxi_id FROM rometaxidata WHERE sim_day_num = %s" % (str(sim_day_num)))
#taxi_ids = pdsql.read_sql_query(execution_str,connection)


"""
bunch of taxi ids on day 3
129
195
106
120
285
8
264
305
318
179
209
276
"""



#taxi_id = 129 #129 #taxi_ids['taxi_id'][0]



#execution_str = ("SELECT unix_ts,lat1,long1 FROM rometaxidata WHERE (taxi_id = %s AND sim_day_num = %s)" % (str(taxi_id),str(sim_day_num)))


connect_str = "dbname='c207rometaxitraces' user='postgres' host='localhost' password='postgres'"

sim_day_num = 4

connection = psycopg2.connect(connect_str)


#1. get all taxi trace data for one day.
execution_str = ("SELECT taxi_id,unix_ts,lat1,long1 FROM rometaxidata WHERE sim_day_num =%s" % (str(sim_day_num)))
taxidf = pdsql.read_sql_query(execution_str,connection)


# List unique values in a DataFrame column
# h/t @makmanalp for the updated syntax!
taxi_IDs = list(taxidf['taxi_id'].unique())


# Grab DataFrame rows where column has certain values
#valuelist = ['value1', 'value2', 'value3']
#df = df[df.column.isin(valuelist)]



for j in range(0,len(taxi_IDs)):

    trace_data2match = taxidf[taxidf['taxi_id']==taxi_IDs[j]]

    trace_data2match = trace_data2match.sort_values('unix_ts') #VERY IMPORTANT for osrm. big deal!
    #search_radius = np.zeros_like(np.array(taxidf['unix_ts']))+10
    #time_stamps = taxidf['unix_ts']

    #going back to shitty python wrapper:
    m = 0#50 #between 50-60 there is an error... the timestamps are not monotonically increasing.. need to sort this, jokes.
    n = 1260 #60 #900 #len(gps_subset) #1260
    #mpmatched_points = osrm.match(gps_positions[m:n], overview="simplified", timestamps=taxidf['unix_ts'][m:n], radius=None)

    #bear in mind... i might need to flip lats/longs order... hmmm....
    gps_subset = trace_data2match[['long1','lat1']]
    gps_positions = [tuple(x) for x in gps_subset.values]
    mpmatched_points = osrm.match(gps_positions, overview="simplified", timestamps=trace_data2match['unix_ts'], radius=None)

    nobody_index = []
    matched_longitude = []
    matched_latitude = []
    matched_unix_ts = []

    matched_cols = ['taxi_id','day_num','unix_ts','mlatitude','mlongitude']

    for i in range(0,len(mpmatched_points['tracepoints'])):

        if mpmatched_points['tracepoints'][i] is None:
            nobody_index.append(i)
        else:
            matched_unix_ts.append(taxidf['unix_ts'][i])
            matched_longitude.append(mpmatched_points['tracepoints'][i]['location'][1])
            matched_latitude.append(mpmatched_points['tracepoints'][i]['location'][0])

    matched_taxi_id = np.ones_like(matched_unix_ts)*taxi_IDs[j]
    matched_day_num = np.ones_like(matched_taxi_id)*sim_day_num
    matched_df = pd.DataFrame(np.column_stack([matched_taxi_id,matched_day_num,matched_unix_ts, matched_longitude, matched_latitude]), columns = matched_cols)

    matched_df.taxi_id = matched_df.taxi_id.astype(int)
    matched_df.day_num = matched_df.day_num.astype(int)
    matched_df.unix_ts = matched_df.unix_ts.astype(int)


    if j>0:
            entire_day_matched_traces = pd.concat([entire_day_matched_traces, matched_df], axis=0, join='outer', join_axes=None, ignore_index=True,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True)
    else:
        entire_day_matched_traces = matched_df



#con.execute('TRUNCATE matchedta ;')
#df.to_sql('my_table', con, if_exists='append')

#connect_str2 = "dbname='matchedtaxitraces' user='postgres' host='localhost' password='postgres'"

#connection2 = psycopg2.connect(connect_str2)
#execution_str2 = ("TRUNCATE matchedtaxidata;") #really stupid code.
#matched_df.to_sql(execution_str2,connection2)


# code insert to postgres table, but first, set up table....

#could numpy row stack, and write to database at the end of the 'sim_day_num'... would reduce some shiz...
# similarily, I should import a days worth of traces, divide into unique taxi_ids, then iterate!





''' my attempt...

0. will need to sort out matched database table etc....
this might mean care sigfigs etc... 

1. load maybe 1GB a time from psql...
	2. chunk it up, 
	per chunk
	- psql query (yeah it will be slower, deal with it.... lets get this nigger up and running,)
	- map match: 1000 points? <-- look at above not regards 'gaps=false?'? maybe need to edit fucking pyosrm shit.
	- convert results to
			- pandas dataframe, with ['unix_ts','latitude','longitude'] <-- ORDER IS IMPORTANT BE CAREFUL.
			- save 'overview=full' json file (although writing this out for everything coul be slooow)
			or could I save this to yet another database... nah, easy, just save to disk in another directory
			should have IterationNum,sim_day_num_taxi_id
            - save pandas dataframe to new matched_db




search_radius = [20,20,20,20,20]
url0 = ['http://localhost:5000/match/v1/driving/']




overview = 'full'
steps='false'
geometry='polyline'
gps_points2match = gps_positions[500:1000]
timestamps = taxidf['unix_ts'][500:1000]




#url1 = [url0,';'.join([','.join([str(coord[0]),str(coord[1])]) for coord in gps_points2match])]
#url2 = [join([url0,';'.join([','.join([str(coord[0]),str(coord[1])]) for coord in gps_points2match])])]

#GPS coords...
url0.append(';'.join([','.join([str(coord[0]),str(coord[1])]) for coord in gps_points2match]))
#radiuses
url0.append(';'.join([','.join([str(radii)]) for radii in search_radius]))

url0.append(';'.join([','.join([str(ts)]) for ts in timestamps]))


#timestamps//

url1 = ''.join([url0[0],url0[1]])
url1 = ''.join([url1,'?overview={}&steps={}&geometries={}'.format(overview,str(steps).lower(), geometry)])
#url1 = '&radiuses='.join([url1,url0[2]]) # bold.
url1 = '&timestamps='.join([url1,url0[3]])



url2txt_file = open("url2test.txt","w")
url2txt_file.write(url1)
url2txt_file.close()

#url2txt = np.array(url1
#url_filename = '/home/user/RomeTaxiData/url2test.txt'
#np.savetxt(url_filename,url2txt,fmt=str)


#url = [host, '/match/', url_config.version, '/', url_config.profile, '/',';'.join([','.join([str(coord[0]), str(coord[1])]) for coord in points]),"?overview={}&steps={}&geometries={}".format(overview,str(steps).lower(), geometry)]

'''

#mpmatched_points = osrm.match(gps_positions[0:5], overview="full", timestamps=taxidf['unix_ts'][0:5], radius =[10])


#---- from osrm-python wrapper ----
"""
next steps:
1. edit/start from scratch making new python-osrm map-match function, need to get that url query just right

http://localhost:5000/match/v1/driving/{gps points long1,lat1;... longN,latN}&radiuses={r1;r2;r3...rN}&timestamps{ts1;ts2;ts3...tsN}

url = 'http://localhost:5000/match/v1/driving/'

gps_points2match = gps_positions[500:505]
for i in gps_points2match:
	url = [url.join(str(coord[0]),str(coord[1]) for coord in gps_points2match]


#-------original.... shit.

points = gps_positions

#    host = check_host(url_config.host)

url = [host, '/match/', url_config.version, '/', url_config.profile, '/',';'.join(
            [','.join([str(coord[0]), str(coord[1])]) for coord in points]),
        "?overview={}&steps={}&geometries={}"
           .format(overview, str(steps).lower(), geometry)]

    if radius:
        url.append(";".join([str(rad) for rad in radius]))
    if timestamps:
        url.append(";".join([str(timestamp) for timestamp in timestamps]))

    r = urlopen("".join(url))
    r_json = json.loads(r.read().decode('utf-8'))




for taxi_id in list_taxi_ids:
	
	execution_str = ("SELECT lat1,long1,unix_ts FROM rometaxidata WHERE taxi_id = %s" % (str(taxi_id))
MOAR NOTES

these two seem to work reasonably well, however give slightly different results which may/may not be worrying...

curl "http://router.project-osrm.org/match/v1/driving/12.457089,41.895786;12.457089,41.895786;12.487011,41.893273;12.498969,41.902191;12.501389,41.901612?radiuses=20;20;20;20;20&timestamps=1391371881;1391371882;1391372422;1391372747;1391372791"

curl "http://localhost:5000/match/v1/driving/12.457089,41.895786;12.457089,41.895786;12.487011,41.893273;12.498969,41.902191;12.501389,41.901612?radiuses=20;20;20;20;20&timestamps=1391371881;1391371882;1391372422;1391372747;1391372791"

--



"""
