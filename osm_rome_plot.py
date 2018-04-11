# matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

fig, ax = plt.subplots(figsize=(10,15))

m = Basemap(
	resolution = 'c',
	projection = 'merc',
	llcrnrlon=12.442, llcrnrlat= 41.856, urcrnrlon=12.5387, urcrnrlat= 41.928)

#(12.442 41.856, 12.5387 41.928, 12.442 41.928, 12.5387 41.856, 12.442 41.856)

#m.fillcontinents(color='#f0f0f0')

#road_shp_file = '/home/pdawg/RomeOSMdata/extract/extracted_Roman_roads'

m.readshapefile('extracted_Roman_roads', 'roads', drawbounds = True, color='grey')

plt.savefig('/home/pdawg/RomeTaxiData/rome_road_network.png')




#extract relevant trace data
#extract relevant osm map data
#plot both..
