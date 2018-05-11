# https://gis.stackexchange.com/questions/127878/line-vs-polygon-intersection-coordinates

#https://gis.stackexchange.com/questions/95670/how-to-create-a-shapely-linestring-from-two-points

#http://toblerity.org/shapely/manual.html
#note, the .shp file appears to be of the order: longitude, latitude


from shapely.geometry import Point, LineString, shape
import fiona

#import shapefile as shp

#nigb location:
#poly_file = fiona.open('/home/elizabeth/Downloads/rome_only_osm_data/shp_city_rome/shape/buildings.shp')

#C207
#poly_file = fiona.open('/home/user/Downloads/rome_only_osm_data/shp_city_rome/shape/buildings.shp')

#s_poly = shape(poly_file.next()['geometry'])
#s_line = LineString([Point(12.497626,41.897156), Point(12.497732,41.897516)]).wkt
s_line = LineString([(12.497626,41.897156),(12.497732,41.897516)])
s_line =  LineString([(12.497626,41.897156),(12.4922,41.8902)])
#Multi_pol_ext = MultiLineString([list(shape(pol['geometry']).exterior.coords) for pol in poly_file])
# clearly this does not work...
#print(s_poly.intersection(s_line))

"""
plt.figure()
sf = shp.Reader('/home/user/Downloads/rome_only_osm_data/shp_city_rome/shape/buildings.shp')
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)
plt.show()
"""
#https://gis.stackexchange.com/questions/227423/how-to-efficiently-determine-which-of-thousands-of-polygons-intersect-with-a-lin

import geopandas as gp
from geopandas.tools import sjoin

poly = gp.GeoDataFrame.from_file('/home/user/Downloads/rome_only_osm_data/shp_city_rome/shape/buildings.shp')
lines = gp.GeoDataFrame(geometry = [s_line])

intersections = gp.sjoin(poly, lines, how="inner", op='intersects')
print(intersections)
#this works but is super fucking slow.

#now for some checking, good to plot some of this fucking shit...
from matplotlib.patches import Polygon as mpl_Polygon
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
import pysal as ps
import matplotlib.pyplot as plt
import numpy as np

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

s_line =  LineString([(12.497626,41.897156),(12.4922,41.8902)])
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




