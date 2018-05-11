import urllib2
#params = {'xmin': -6.185440796831979, 'ymin': 106.82374835014343,'xmax': -6.178966266481431, 'ymax': 106.83127999305725}
params = {'xmin': 12.4420, 'ymin':41.8560, 'xmax':12.5387, 'ymax':41.9280}
myOsmXmlUrlPath = ('http://overpass-api.de/api/interpreter?data=(node[%%22building%%22=%%22yes%22](%(xmin)s,%(ymin)s,%(xmax)s,%(ymax)s);way[%22building%22=%22yes%22](%(xmin)s,%(ymin)s,%(xmax)s,%(ymax)s);relation[%22building%22=%22yes%22](%(xmin)s,%(ymin)s,%(xmax)s,%(ymax)s););(._;%3E;);out%20body;' % params)
# Note that osm json is NOT geojson!
myOsmJsonUrlPath = ('http://overpass-api.de/api/interpreter?data=[out:json];(node[%22building%22=%22yes%22](%(xmin)s,%(ymin)s,%(xmax)s,%(ymax)s);way[%22building%22=%22yes%22](%(xmin)s,%(ymin)s,%(xmax)s,%(ymax)s);relation[%22building%22=%22yes%22](%(xmin)s,%(ymin)s,%(xmax)s,%(ymax)s););(._;%3E;);out%20body;' % params)
myRequest = urllib2.Request(myOsmXmlUrlPath)
try:
    myUrlHandle = urllib2.urlopen(myRequest, timeout=60)
    myFile = file('osm.xml', 'wb')
    myFile.write(myUrlHandle.read())
    myFile.close()
