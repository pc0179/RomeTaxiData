# pc0179 on the Kboard, python 3.5+ throughout.
# osrm version v1.18 
# osrm-routed ? -a CH --max-table-size 250

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

#for Line-of-Sight Model
from shapely.geometry import Point, LineString, shape, mapping, MultiLineString
import fiona
import geopandas as gp
from geopandas.tools import sjoin

#from matplotlib.patches import Polygon as mpl_Polygon
#from matplotlib.collections import PatchCollection
#from descartes import PolygonPatch
#import pysal as ps

#some useful-ish functions:
def Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T):
# strictly moving from x1,y1 --> x2,y2,...
# t1<t2.
    dt = t2-t1
    dT = T-t1
    xT = dT*(x2-x1)/dt + x1
    yT = dT*(y2-y1)/dt + y1

    return round(xT,6),round(yT,6)

def haversine_dist_matrix(latitudes,longitudes):
    
    x = np.asarray(longitudes)
    m, k = x.shape
    y = np.asarray(latitudes)
    n, kk = y.shape
    result = np.empty((m,n),dtype=float)

    if m<n:
        for i in range(m):
           result[i,:] = spherical_dist(x[i],y)
    else:
        for j in range(n):
            result[:,j] = spherical_dist(x,y[j])

    return result

def spherical_dist(pos1, pos2, r=3958.75e3):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

def Route_Osrm(start_long,start_lat,target_long,target_lat,start_timestamp,target_timestamp):

    route_result = osrm.simple_route([start_long,start_lat],[target_long,target_lat],output='full',overview="full", geometry='polyline',steps='True',annotations='true')

    encoded_polyline = route_result['routes'][0]['geometry']

#plot_route_nodes = pd.DataFrame(polyline.decode(encoded_polyline))

    route_nodes = pd.DataFrame(polyline.decode(encoded_polyline), columns=['lat','long'])

    link_data = pd.DataFrame({'distance':route_result['routes'][0]['legs'][0]['annotation']['distance'], 'duration': route_result['routes'][0]['legs'][0]['annotation']['duration'], 'dur_cumsum' : (((np.cumsum(route_result['routes'][0]['legs'][0]['annotation']['duration'])/np.sum(route_result['routes'][0]['legs'][0]['annotation']['duration']))*(target_timestamp-start_timestamp))+start_timestamp)})
# this last section, allows for temporal scaling, i.e. if the start and end times don't match what
# osrm predicts for the journey, the real time of the taxi is split in proportion to the line segments predict journey time by osrm.

    return link_data, route_nodes

def Straight_Line_Distance(x1,y1,x2,y2):
    d = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return d

#~--- now for some serious scripting.........................................
#1. Query taxi trace database for trace data at a particular time, T

#Time of interest:
T_date = dt.datetime(2014,2,3,12,30,25)
T_unix = int(T_date.timestamp())
t_margin = 30 #in seconds..., i.e. a minute eitherside??? note taxi_id = 8 produces interesting...
t_mini_margin = 2 # seconds either side, where we accept the current result

#Connection to database
# for C207:
#connect_str = "dbname='rometaxitraces' user='postgres' host='localhost' password='postgres'"
#fur klara:
#connect_str = "dbname='c207rometaxitraces' user='postgres' host='localhost' password='postgres'"
#fur NiGB,
#connect_str = "dbname='nigb_romedata' user='postgres' host='localhost' password='postgres'"
#fur mike_pc
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)

# fur everything else:
#execution_str = ("SELECT taxi_id,unix_ts,lat1,long1,x,y FROM rometaxidata WHERE unix_ts BETWEEN %s AND %s " % (str(T_unix-t_margin),str(T_unix+t_margin)))

#fur mike-pc
execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude,x,y FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T_unix-t_margin),str(T_unix+t_margin)))

#2. Quick filter, db data to pandas dataframe
taxidf = pdsql.read_sql_query(execution_str,connection)
taxidf = taxidf.drop_duplicates() #removes duplicates, an ongoing problem.


# taxis within t_mini_margin, are assumed to be correct, no further processing here
prime_taxis = taxidf.loc[(taxidf.unix_ts>T_unix-t_mini_margin) & (taxidf.unix_ts<T_unix+t_mini_margin)]

estimate_taxis_positions = pd.DataFrame({'latitude':list(prime_taxis.latitude),'longitude':list(prime_taxis.longitude)}, index=list(prime_taxis.taxi_id))


# add diff column... (unix_ts - user set time, T)
taxidf2 = taxidf.drop(taxidf[(taxidf.unix_ts>T_unix-t_mini_margin) & (taxidf.unix_ts<T_unix+t_mini_margin)].index)

taxidf2['ts_diff'] = taxidf2.unix_ts-T_unix

beforedf = taxidf2[taxidf2['unix_ts']<T_unix]
afterdf = taxidf2[taxidf2['unix_ts']>T_unix]

before_taxi_ids = list(beforedf.taxi_id.unique())
after_taxi_ids = list(afterdf.taxi_id.unique())
taxi_ids = set(before_taxi_ids).intersection(after_taxi_ids)

#3. loop through all taxi_id's estimating their position at chosen time, T
apprx_txi_lat1 = []
apprx_txi_long1 = []
apprx_txi_id = []

for taxi_id in taxi_ids:

    bdf = beforedf[beforedf['taxi_id']==taxi_id] #taxi_ids[8]]
    adf = afterdf[afterdf['taxi_id']==taxi_id] #taxi_ids[8]]
    if (len(adf)+len(bdf))>2:
		
        bdf2 = bdf.loc[bdf['ts_diff'].idxmax()]
        adf2 = adf.loc[adf['ts_diff'].idxmin()]
        # maybe somewhere about here, need to make some checks
        # if distance between chosen points is <D_min: pick whatever is closest
        # else: do some routing and interping between points...
        d = Straight_Line_Distance(adf2.x,adf2.y,bdf2.x,bdf2.y)

        D_min = 15 # minimum distance in metres actually worth interpolating...
        if d<D_min:
    #taxi_position = [bdf2.long1,bdf2.lat1] #maybe in future use nearest value...
            apprx_txi_lat1.append(bdf2.latitude)
            apprx_txi_long1.append(bdf2.longitude)
            apprx_txi_id.append(taxi_id)


        else: 
            link_data, route_nodes  = Route_Osrm(bdf2.longitude,bdf2.latitude,adf2.longitude,adf2.latitude,bdf2.unix_ts,adf2.unix_ts)

            a = link_data[link_data.dur_cumsum>T_unix].dur_cumsum.idxmin()
	# herein at a, lies an interesting problem. There are occassions where the vehicle travels exceedingly slowly, therefore it's start and end time  do not match the times predicted by osrm (to get from A-->B), i.e. osrm_generated_timestamp_at_B < actual timestamp at b, therefore need to think about scaling, such that the duration cumsum takes into account the discrepency between actuall_end_timestamp and the one predicted by osrm.
            b = a+1

            x1 = route_nodes['long'][a]
            y1 = route_nodes['lat'][a]
            if a==0:
	            t1 = link_data['dur_cumsum'][a]-link_data.duration[a]
            else:
	            t1 = link_data['dur_cumsum'][a-1]

            x2 = route_nodes['long'][b]
            y2 = route_nodes['lat'][b]
            t2 = link_data['dur_cumsum'][a]


            xT,yT = Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T_unix)

            apprx_txi_lat1.append(yT)
            apprx_txi_long1.append(xT)
            apprx_txi_id.append(taxi_id)



#4. results from loop, estimate linear distance between taxis for initial sorting...
# regards Line of Sight (LoS) model
apprx_txi_pos_df = pd.DataFrame({'latitude':apprx_txi_lat1,'longitude':apprx_txi_long1}, index=apprx_txi_id) #uses taxi id as index... could be interesting

estimate_taxis_positions = estimate_taxis_positions.append(apprx_txi_pos_df)

# plots last point of iteration,
#plt.figure()
#plt.plot(route_nodes['long'],route_nodes['lat'],'+-k')
#plt.plot(x1,y1,'dg')
#plt.plot(x2,y2,'dr')
#plt.plot(xT,yT,'ob') #show be inbetween other two points.
#plt.show()


#5. Line of Sight Model

linear_dist_df = pd.DataFrame(haversine_dist_matrix(estimate_taxis_positions.values, estimate_taxis_positions.values),index=estimate_taxis_positions.index, columns=estimate_taxis_positions.index)
linear_dist_mat = np.array(haversine_dist_matrix(estimate_taxis_positions.values, estimate_taxis_positions.values))


#carzy route_dist vs linear_dist..., need to ensure shortest distance is picked for comparison.
"""table_input_df_cols = ['long1','lat1'] #change the order, long,lat for table query input
estimate_taxis_positions = estimate_taxis_positions[table_input_df_cols]
table_input_coords = estimate_taxis_positions.values.tolist()
route_table_index = estimate_taxis_positions.index.tolist()

route_dist_df = osrm.table(table_input_coords, route_table_index, output='dataframe', send_as_polyline=True, annotations='distance')


queck = np.where((linear_dist_mat>0)&(linear_dist_mat<150)) #returns tuple of two arrays, each array is row /column index...


#this LoS model works but is super fucking slow.


s_line =  LineString([(12.497626,41.897156),(12.4922,41.8902)])

poly = gp.GeoDataFrame.from_file('/home/user/Downloads/rome_only_osm_data/shp_city_rome/shape/buildings.shp')
lines = gp.GeoDataFrame(geometry = [s_line])

intersections = gp.sjoin(poly, lines, how="inner", op='intersects')
print(intersections)

test_los_line = LineString([(,)
shapefile = '/home/user/Downloads/rome_only_osm_data/shp_city_rome/shape/buildings.shp'

df_map_elements = gp.GeoDataFrame.from_file(shapefile)

df_map_elements["mpl_polygon"] = np.nan
df_map_elements['mpl_polygon'] = df_map_elements['mpl_polygon'].astype(object)
for self_index, self_row_df in df_map_elements.iterrows():
    m_polygon = self_row_df['geometry']
    poly=[]
    if m_polygon.geom_type == 'MultiPolygon':
        for pol in m_polygon:
            poly.append(PolygonPatch(pol))
    else:
        poly.append(PolygonPatch(m_polygon))
    df_map_elements.set_value(self_index, 'mpl_polygon', poly)

dict_mapindex_mpl_polygon = df_map_elements['mpl_polygon'].to_dict()


x_line = [12.497626,12.4922]
y_line = [41.897156,41.8902]

#plt.show()

fig, ax = plt.subplots()
for c_l ,patches in dict_mapindex_mpl_polygon.items():
    p = PatchCollection(patches,color='white',lw=.3,edgecolor='k')
    ax.add_collection(p)
ax.autoscale_view()
plt.plot(x_line,y_line,'-*r',lw=2)
plt.show()


"""

#speed up attempt 1, use postgis database with Rome bulidings shapefile loaded and indexed


# RIGHT. TO DO before meeting, need to set up rome building shapefile database on KLARA and
# query it to test LoS for each of the elements of 'queck', which are the list of taxis 
# within x metres from each other....


queck = np.where((linear_dist_mat>0)&(linear_dist_mat<200))

def MiniLinearDistFilter(linear_dist_mat,min_los_length,max_los_length):
    queck = []
    for row in range(0,len(linear_dist_mat)):
        for col in range(0,row):
            if (linear_dist_mat[row][col]> min_los_length) & (linear_dist_mat[row][col]< max_los_length):
                queck.append([row,col])

    return queck

# for loop would start here, for now, just two taxis at a time,
# might consider sending this all in LineStrings to DB and query it all in one go from there for even faster speed up...
taxi_a = estimate_taxis_positions.iloc[queck[0][0]].tolist()
taxi_b = estimate_taxis_positions.iloc[queck[1][0]].tolist()

#Connection to database
# for KLARA,,,,,:
#LoS_connect_str = "dbname='shp_rome_klara' user='postgres' host='localhost' password='postgres'"
#LoS_connection = psycopg2.connect(LoS_connect_str)

#just for trial purposes... validataion....
# s_line =  LineString([(12.497626,41.897156),(12.4922,41.8902)]) #should produce 11, i.e 11 buildings in the way....
taxi_a = [12.497626,41.897156]
taxi_b = [12.4922,41.8902]
LoS_execution_str = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxi_a[0]),str(taxi_a[1]),str(taxi_b[0]),str(taxi_b[1])))

LoS_df1 = pdsql.read_sql_query(LoS_execution_str,connection)


LoS_results = []
los_line_string_list = []
los_line_list = []
nolos_line_string_list = []
nolos_line_list = []

min_los_length = 0
max_los_length = 200
nearby_taxis = MiniLinearDistFilter(linear_dist_mat,min_los_length,max_los_length)

for i in range(0,len(nearby_taxis)):
#for i in range(0,len(queck[0])):
    
    #taxi_a = estimate_taxis_positions.iloc[queck[0][i]].tolist()
    #taxi_b = estimate_taxis_positions.iloc[queck[1][i]].tolist()
    
    taxi_a = estimate_taxis_positions.iloc[nearby_taxis[i][0]].tolist()
    taxi_b = estimate_taxis_positions.iloc[nearby_taxis[i][1]].tolist()
    
    LoS_execution_str = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxi_a[1]),str(taxi_a[0]),str(taxi_b[1]),str(taxi_b[0])))

    LoS_df = pdsql.read_sql_query(LoS_execution_str,connection)
    
    LoS_results.append(len(LoS_df))


    if LoS_results[-1]>0:
        nolos_coords = [(taxi_a[1],taxi_a[0]),(taxi_b[1],taxi_b[0])] #yeah switched for some weird reason.
      
        nolos_line_string_list.append(LineString(nolos_coords))
        nolos_line_list.append(nolos_coords)
    else:
        los_coords = [(taxi_a[1],taxi_a[0]),(taxi_b[1],taxi_b[0])] #yeah switched for some weird reason.
        los_line_string_list.append(LineString(los_coords))
        los_line_list.append(los_coords)



# writing to shapefile, for easier plotting...
#multicoords = [line_list for line in line_string_list]
# Making a flat list -> LineString

def Save_LoS_Shpfile(line_list,shp_file_name):

    schema = {'geometry': 'LineString','properties': {'id': 'int'}}
    with fiona.open(('%s.shp' % (shp_file_name)), 'w', 'ESRI Shapefile', schema)  as output:
    
        for i in range(0,len(line_list)):
            simple3 = LineString(line_list[i])
            output.write({'geometry':mapping(simple3),'properties': {'id':1}})

"""
def Save_LoS_shpfile(line_list,shp_file_name):

    simple = LineString([item for sublist in line_list  for item in sublist])
    simple2 = MultiLineString([item for sublist in line_list  for item in sublist])
# resulting shapefile

   aschema = {'geometry': 'LineString','properties': {'id': 'int'}}
    with fiona.open(('%s.shp' % (shp_file_name)), 'w', 'ESRI Shapefile', schema)  as output:
       output.write({'geometry':mapping(simple2),'properties': {'id':1}})
"""

shp_file_name = ['los_rome_taxi_lines.shp','nolos_rome_taxi_lines.shp']
Save_LoS_Shpfile(los_line_list,shp_file_name[0])
Save_LoS_Shpfile(nolos_line_list,shp_file_name[1])



