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

route_result = osrm.simple_route([12.442,41.856],[12.450,41.801],output='full',overview="full", geometry='polyline',steps='True',annotations='true')

encoded_polyline = route_result['routes'][0]['geometry']

route_nodes = polyline.decode(encoded_polyline)

link_lengths = route_result['routes'][0]['legs'][0]['annotation']['distance']

link_times = route_result['routes'][0]['legs'][0]['annotation']['duration']

# very clumsy interpolation attempt
T = 80
node_ix = len(cumsum_times[cumsum_times<T])

#slap following into my fast linearinterp machine... get gps points at precisely T=t
routes_nodes[node_ix]

route_nodes[node_ix-1]




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




