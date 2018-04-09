import _mysql as mysqlc

db = mysqlc.connect(host = "", user="root", passwd="ToshibaTRL37",db="rometaxi")

cur = db.cursor()

cur.execute("SELECT * FROM test_trace")

for row in cur.fetchall():
	print(row[0], " ", row[1])

