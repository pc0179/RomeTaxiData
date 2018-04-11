# matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

fig, ax = plt.subplots(figsize=(10,15))

m = Basemap(
	resolution = 'c',
	projection = 'merc',
	llcrnrlon=-99.22, llcrnrlat= 19.33, urcrnrlon=-99.12, urcrnrlat=19.45)

#m.fillcontinents(color='#f0f0f0')

m.readshapefile('extracted_Roman_roads', 'roads', drawbounds = True, color='grey')

plt.savefig('rome_road_network.png')
