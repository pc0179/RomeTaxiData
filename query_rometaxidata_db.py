import psycopg2
import csv


connect_str = "dbname='pdawg' user='pdawg' host='localhost' password='ToshibaTRL37'"
# quick reminder of available columns in database:
# cols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','Lat1','Long1','x','y']

conn = psycopg2.connect(connect_str)
cursor = conn.cursor()

#interesting queries...
#"SELECT * FROM rometaxidata WHERE taxi_id =225"
execution_str = "SELECT * FROM rometaxidata WHERE taxi_id =129 AND sim_day_num=11"

#execution_str = "SELECT * FROM rometaxidata WHERE sim_day_num=11"
#execution_str = "SELECT * FROM rometaxidata WHERE (x BETWEEN -1000 AND 1000) AND (y BETWEEN -1000 AND 1000)"
#execution_str = "SELECT * FROM rometaxidata WHERE sim_day_num = 10 AND (x BETWEEN -1000 AND 1000) AND (y BETWEEN -1000 AND 1000)" 
#execution_str = "SELECT * FROM rometaxidata WHERE weekday_num = 0 AND taxi_id = 225"
#execution_str = "SELECT DISTINCT taxi_id FROM rometaxidata"

#execute the fucking query like its a french royal:
cursor.execute(execution_str)

psql_rows_out = cursor.fetchall()
#Output query results to csv file:
with open("psql_query_output.csv", "w", newline='') as f:
	writer = csv.writer(f)
	writer.writerows(psql_rows_out)


