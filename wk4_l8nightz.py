import osrm
import numpy as np
import pandas as pd
import polyline



# a slow snap back to reality... 
# need to query osrm server, for route between points, then 
# interpoloate between them, to find, once again, the answer to the
# the enternal question: where is the taxi at time T?

# test set, traverse this bitch; (41.856, 12.442),(41.928,12.5387)

# Do note however, that NiGB's version of osrm, is not the MLD but the CH routing algo.

#route = osrm.simple_route([12.442,41.856],[12.5387,41.928],output='full',overview="full", geometry='polyline',steps='True')


def Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T):
# strictly moving from x1,y1 --> x2,y2,...
# t1<t2.

	dt = t2-t1
	dT = T-t1
	xT = dT*(x2-x1)/dt + x1
	yT = dT*(y2-y1)/dt + y1

	return round(xT,6),round(yT,6)



route_result = osrm.simple_route([12.442,41.856],[12.5387,41.928],output='full',overview="full", geometry='polyline',steps='True',annotations='true')

encoded_polyline = route_result['routes'][0]['geometry']

plot_route_nodes = pd.DataFrame(polyline.decode(encoded_polyline))

route_nodes = polyline.decode(encoded_polyline)
#link_lengths = route_result['routes'][0]['legs'][0]['annotation']['distance']

#link_times = route_result['routes'][0]['legs'][0]['annotation']['duration']

link_data = pd.DataFrame({'distance':route_result['routes'][0]['legs'][0]['annotation']['distance'], 'duration': route_result['routes'][0]['legs'][0]['annotation']['duration'], 'dur_cumsum' : np.cumsum(route_result['routes'][0]['legs'][0]['annotation']['duration'])})
 
# very clumsy interpolation attempt
# continuing this line of thought...
# in theory, it starts with a query to db: select tax_id, lat1, long1, t, NEAREST Group by taxi_id?
# if near enough points.. go for it, otherwise need to group by taxi_id, and process each 
# individual trace, ie, the two nearest points to time T
# ie query result = taxi_id_I, x1,y1,t1 and x2,y2,t2... for all taxi_ids within range of query

# processing, requires (map-matching?---ssshh!) then route between points, interpolate, find best estimate
T_query = 1525082338+400 #eg some unix time or date-time obj.
t1 = 1525082338 # earliest poitn, t1<t2... dummy number, actuall number will be returned from initial point (x1,y1,t1)

T_query = 1020.2
t1 = 0
link_data['dur_cumsum'] = link_data['dur_cumsum'] + t1

interp_points_index = link_data.loc[:,'dur_cumsum']>T_query

# actually a bi useless when you think about it: a = interp_points_index[~interp_points_index ].index[-1]
a = interp_points_index[interp_points_index ].index[0]
b = a+1

x1 = route_nodes[a][0]
y1 = route_nodes[a][1]
t1 = link_data['dur_cumsum'][a-1]

x2 = route_nodes[b][0]
y2 = route_nodes[b][1]
t2 = link_data['dur_cumsum'][a]

xT,yT = mid_point(x1,y1,t1,x2,y2,t2,T_query)
#xT,yT = mid_point(route_nodes[a][0],route_nodes[a][1],link_data['dur_cumsum'][a],route_nodes[b][0],route_nodes[b][1],link_data['dur_cumsum'][b], T_query)

import matplotlib.pyplot as plt

plt.plot(plot_route_nodes[0],plot_route_nodes[1],'+-k')
plt.plot(x1,y1,'dg')
plt.plot(x2,y2,'dr')
plt.plot(xT,yT,'ob') #show be inbetween other two points.
plt.show()


#https://router.project-osrm.org/route/v1/driving/polyline(__n~Foa%7DjA_aMk%7BQ)?overview=full&steps=false&alternatives=false&geometries=polyline&annotations=distance,speed,duration

#in theory, the duration unit is in seconds, since I know (x1,y1,t1) and (x2,y2,t2) I  could just cum-sum this bitch up, 

# well yes indeed, but upon further inspection,  it seems hard to find where along a link the taxi might be since the route service isn't returning gps coordinates of points along route... just some damn node ids.

# hang on...
# hidden deep in the summary section, there appears to be gps points of every intersection (i.e. I presume node) of the route traversed
# decode the polyline!! it has lat longs to 5/6 dp's.



# the plot thickens,... a = route[0]['legs'] might have all the necessary data to perform interpolations?
# import polyline
# b = polyline.decode(encoded_polyline)
# ~= route['routes'][0]['geometry']
# encoded_polyline = route['routes'][0]['geometry']
# route_nodes = polyline.decode(encoded_polyline)

# for link lengths?
# route['routes'][0]['legs'][0]['annotation']




