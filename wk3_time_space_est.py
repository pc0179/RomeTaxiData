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

connect_str = "dbname='rometaxitraces' user='postgres' host='localhost' password='postgres'"

taxi_ids = pd.read_csv('/home/user/RomeTaxiData/all_rome_taxi_ids.csv', header=None, sep="\n")
list_taxi_ids = list(taxi_ids[0]) #I was bored and this seemed easier than figuring out exactly how pandas.iterrows() bullshit works

osrm.RequestConfig.host = "http://localhost:5000"

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

connection = psycopg2.connect(connect_str)

#execution_str = "SELECT DISTINCT taxi_id FROM rometaxidata"

execution_str = "SELECT lat1,long1,unix_ts FROM rometaxidata WHERE taxi_id = 225"

taxidf = pdsql.read_sql_query(execution_str,connection)

#bear in mind... i might need to flip lats/longs order... hmmm....
gps_subset = taxidf[['long1','lat1']]
gps_positions = [tuple(x) for x in gps_subset.values]
search_radius = np.zeros_like(np.array(taxidf['unix_ts']))+10
#time_stamps = taxidf['unix_ts']

#search_radius = [20,20,20,20,20]
url0 = ['http://localhost:5000/match/v1/driving/']




overview = 'full'
steps='false'
geometry='polyline'
gps_points2match = gps_positions #[500:505]
timestamps = taxidf['unix_ts'] #[500:505]




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
url1 = '&radiuses='.join([url1,url0[2]]) # bold.
url1 = '&timestamps='.join([url1,url0[3]])



url2txt_file = open("url2test.txt","w")
url2txt_file.write(url1)
url2txt_file.close()

url2txt = np.array(url1
url_filename = '/home/user/RomeTaxiData/url2test.txt'
np.savetxt(url_filename,url2txt,fmt=str)


#url = [host, '/match/', url_config.version, '/', url_config.profile, '/',';'.join([','.join([str(coord[0]), str(coord[1])]) for coord in points]),"?overview={}&steps={}&geometries={}".format(overview,str(steps).lower(), geometry)]

#mpmatched_points = osrm.match(gps_positions[0:5], overview="full", timestamps=taxidf['unix_ts'][0:5], radius=search_radius[0:5])

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
