# Rome Colosseum
# 41.890251 (lat)
# 12.492373 (long)

#DatumLong = 12.492373
#DatumLat = 41.890251

GPS_str = "POINT(41.8967831636848 12.4821987021152)"

import numpy as np
import re
import RomeTaxiGlobalVars as RTGV

#rx.findall("Some example: Jr. it. was .23 between 2.3 and 42.31 seconds")
# maybe this section needs to go to the end, calc cartesian using full digits? then save just the most sig. figs 6ish for table
#lat1 = 41.8836718276551
#long1 = 12.4877775603346

def LatLongConv(GPS_str):
	numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
	rx = re.compile(numeric_const_pattern, re.VERBOSE)
	a = rx.findall(GPS_str)
	lat1 = round(float(a[0]),6)
	long1 = round(float(a[1]),6)
	return tuple([lat1, long1])


def LatConv2(GPS_str):
	numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
	rx = re.compile(numeric_const_pattern, re.VERBOSE)
	a = rx.findall(GPS_str)
	lat1 = round(float(a[0]),6)
	return lat1

def LongConv2(GPS_str):
	numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
	rx = re.compile(numeric_const_pattern, re.VERBOSE)
	a = rx.findall(GPS_str)
	long1 = round(float(a[1]),6)
	return long1

	
def haversine_pc(lon1,lat1,lon2,lat2):
	lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
	dlon = lon2-lon1
	dlat = lat2-lat1
	a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
	c = 2 * np.arcsin(np.sqrt(a))
	Hdistance = 6371e3*c  #working in metres!
	return Hdistance


def XPos_From_Datum2(Lat1,Long1):
	DatumLat = RTGV.DatumLat
	DatumLong = RTGV.DatumLong
	
	x = round(haversine_pc(Long1,DatumLat,DatumLong,DatumLat)) # ,2)
	if Lat1-DatumLat<0:
		#Implies point is South of Datum
		x=-x
	return int(x)

def YPos_From_Datum2(Lat1,Long1):
	DatumLat = RTGV.DatumLat
	DatumLong = RTGV.DatumLong	
	y = round(haversine_pc(DatumLong,Lat1,DatumLong,DatumLat)) #,2)
	if Long1-DatumLong<0:
	#Implies point is WEST of Datum
		y=-y
	return int(y)

	
def Position_From_Datum(latlong_tuple):
#	if DatumLong== None  & DatumLat == None:
#	DatumLong = 12.492373
#	DatumLat = 41.890251
	DatumLat = RTGV.DatumLat
	DatumLong = RTGV.DatumLong
#if type(latlong_tuple)==tuple:
	lat1 = latlong_tuple[0]
	lon1 = latlong_tuple[1]
#	else:
#		lat1 = latlong_tuple
#		lon1 = latlong_tuple
		
	x = round(haversine_pc(lon1,DatumLat,DatumLong,DatumLat)) # ,2)
	y = round(haversine_pc(DatumLong,lat1,DatumLong,DatumLat)) #,2)

	if lat1-DatumLat<0:
	#Implies point is South of Datum
		x=-x
	if lon1-DatumLong<0:
	#Implies point is WEST of Datum
		y=-y
	return tuple([int(x),int(y)])

if __name__=='__main__':

	lat1, long1 = LatLongConv(GPS_str)
	x,y = Position_From_Datum(long1,lat1,DatumLong,DatumLat)
	print(x,y)
	print(lat1,long1)

