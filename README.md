# RomeTaxiData
Rome Taxi Data-set February 2014

We have the following 'columns' in data
- taxiID
- TimeStamp
- GPS

Converting TimeStamp to datetime obj, seconds since sim started, sim day num, weekday
Converting GPS to shortened GPS (6 sig. figs.) and X,Y coordinate system based around the Colosseum in Rome.

due to size of text file, 1.6GB
aim to use data chunking to reduce memory overload
final text file will be larger due to added fields
aim to smash all this into a sql/postgis database...

global variables such as sim start time, datum location are saved in RomeTaxiGlovalVars

