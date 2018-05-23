# hdawg on the Kboard.
# lets do some plotting, raw vs route vs match vs snap.
# investigation time = 1 hour.

#in general:
import numpy as np
import pandas as pd
import datetime as dt

#for routing:
import osrm
import polyline

#for querying taxi trace database:
import psycopg2
import pandas.io.sql as pdsql
from sqlalchemy import create_engine

#for plotting:
import matplotlib.pyplot as plt

#--------------------

# query/import raw taxi trace data
#Connection to database
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)
#Time of interest:
start_time = dt.datetime(2014,2,2,18,00,00)
Tstart_unix = int(start_time.timestamp())

T_search_times = list(range(Tstart_unix,Tstart_unix+(60*60*1),30)) #search for 14 hours? at every 30 seconds, this is a lot of queries... moving on...
T_search_margin = 60*60*1 # i.e. une heure #15 #i.e. thirty second chunks
t_accept = 1 #second either side? just use this value for position


T = Tstart_unix # for now....

execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T-T_search_margin),str(T+T_search_margin))) 

taxidf = pdsql.read_sql_query(execution_str,connection)
taxidf = taxidf.drop_duplicates() #removes duplicates, incase....
#snap co-ordinates to nearest road segment, no intelligence, no map-matching, just snapping.

#for taxi_id in taxidf.taxi_id.unique():
taxi_id = taxidf.taxi_id.unique()[1]

# snapping to road segment
def Snap2Road(longitude,latitude):
#inputs: lists of longitudes and latidutes
#output: lists of snapped coords

    snapped_longitude = longitude
    snapped_latitude = latitude

    for i in range(0,len(snapped_latitude)):
        
        snapped_result = osrm.nearest((snapped_longitude[i],snapped_latitude[i]))
        snapped_longitude[i] = snapped_result['waypoints'][0]['location'][0]
        snapped_latitude[i] = snapped_result['waypoints'][0]['location'][1]
    
    return snapped_longitude, snapped_latitude

#a = [taxidf.latitude[taxidf.taxi_id==255].value,taxidf.longitude[taxidf.taxi_id==255].value]

snapped_long, snapped_lat = Snap2Road(taxidf.longitude[taxidf.taxi_id==taxi_id].tolist(),taxidf.latitude[taxidf.taxi_id==taxi_id].tolist())


# map-matching

#gps_subset = trace_data2match[['long1','lat1']]

gps_subset = taxidf[taxidf.taxi_id==taxi_id].sort_values('unix_ts')

gps_positions = [tuple(x) for x in gps_subset[['longitude','latitude']].values]
timestamps2match = gps_subset.unix_ts.tolist()

#mpmatched_points = osrm.match(gps_positions, overview="simplified", timestamps=gps_subset.unix_ts.tolist(), radius=None)

#atit = polyline.encode(gps_positions,6)

#mpmatched_points = osrm.match(atit, overview="simplified", timestamps=gps_subset.unix_ts.tolist(), radius=None)

matched_points = osrm.match(gps_positions, overview="simplified", timestamps=timestamps2match, radius=None)


def ProcessMapMatchResults(matched_points, timestamps):

    matched_longs = []
    matched_lats = []
    nobody_index = []
    matched_ts = []

    for i in range(0,len(matched_points['tracepoints'])):

        if matched_points['tracepoints'][i] is None:
            nobody_index.append(i)
        else:
            matched_ts.append(timestamps[i])
            matched_longs.append(matched_points['tracepoints'][i]['location'][0])
            matched_lats.append(matched_points['tracepoints'][i]['location'][1])

    return matched_longs, matched_lats, matched_ts, nobody_index

matched_longs, matched_lats, matched_ts, nobody_index = ProcessMapMatchResults(matched_points, timestamps2match)



# routing

route_result = osrm.simple_route(list(gps_positions[0]),list(gps_positions[-1]), coord_intermediate = [gps_positions[100],gps_positions[200],gps_positions[300]],output='full',overview='full',geometry='polyline',steps='True',annotations='true')

encoded_polyline = route_result['routes'][0]['geometry']
route_nodesdf = pd.DataFrame(polyline.decode(encoded_polyline), columns=['latitude','longitude'])


# plots.

plt.ion()
plt.plot(gps_subset.longitude.tolist(),gps_subset.latitude.tolist(),'-*k', label=('raw taxi:%i data' % (taxi_id)))
plt.plot(snapped_long, snapped_lat, '--sr', label='snapped to road')
plt.plot(matched_longs,matched_lats, '-.og', label='matched trace points')
plt.plot(route_nodesdf.longitude.tolist(),route_nodesdf.latitude.tolist(), '--db', label='fastest route')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend(loc='upper left')
plt.show()
plt.savefig(('osrm_comparison_study_taxi_%s.pdf' % (str(taxi_id))), dpi=400)

"""
    coord_intermediate : list of 2-floats list/tuple
        [(x ,y), (x, y), ...] where x is longitude and y is latitude

simple_route(coord_origin, coord_dest, coord_intermediate=None,
                 alternatives=False, steps=False, output="full",
                 geometry='polyline', overview="simplified",
                 url_config=RequestConfig, send_as_polyline=True,annotations='true'):

def Route_Osrm(start_long,start_lat,target_long,target_lat,start_timestamp,target_timestamp):

    route_result = osrm.simple_route([start_long,start_lat],[target_long,target_lat],output='full',overview="full", geometry='polyline',steps='True',annotations='true')

    encoded_polyline = route_result['routes'][0]['geometry']

#plot_route_nodes = pd.DataFrame(polyline.decode(encoded_polyline))

    route_nodes = pd.DataFrame(polyline.decode(encoded_polyline), columns=['lat','long'])

    link_data = pd.DataFrame({'distance':route_result['routes'][0]['legs'][0]['annotation']['distance'], 'duration': route_result['routes'][0]['legs'][0]['annotation']['duration'], 'dur_cumsum' : (((np.cumsum(route_result['routes'][0]['legs'][0]['annotation']['duration'])/np.sum(route_result['routes'][0]['legs'][0]['annotation']['duration']))*(target_timestamp-start_timestamp))+start_timestamp)})
# this last section, allows for temporal scaling, i.e. if the start and end times don't match what
# osrm predicts for the journey, the real time of the taxi is split in proportion to the line segments predict journey time by osrm.

    return link_data, route_nodes


link_data, route_nodes  = Route_Osrm(bdf2.longitude,bdf2.latitude,adf2.longitude,adf2.latitude,bdf2.unix_ts,adf2.unix_ts)



query, import say 1 hour
pick 1 taxi.
raw
map-match
snap
route(A-B)

plot above.
"""
