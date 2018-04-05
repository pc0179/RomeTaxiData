import datetime as DT

#sim_start_time = '2014-02-01 00:00:00'

#csv_time = '2014-02-01 00:00:03.707117+01'

#chopped_csv_time = csv_time[:-3]


def PyTimeConv(csv_time):
	chopped_csv_time = csv_time[:-3]
	py_time_ms = DT.datetime.strptime(chopped_csv_time,'%Y-%m-%d %H:%M:%S.%f')
	py_time = RoundTimeSeconds(py_time_ms)
	return py_time

def RoundTimeSeconds(some_DT_obj):
	if some_DT_obj.microsecond>= 5e5:
		some_DT_obj = some_DT_obj + DT.timedelta(seconds=1)
	return some_DT_obj.replace(microsecond=0)

def SimuTime(some_DT_obj,sim_start_time):
	sim_time_s = (some_DT_obj-sim_start_time).total_seconds()
	sim_daynum = (some_DT_obj-sim_start_time).days()
	sim_weekday = some_DT_obj.weekday()
	return sim_time_s,sim_daynum,sim_weekday

# in terms of all things related to time, we should end with the following:
# actual_time (datetime obj), sim_time in seconds, sim_time in days, weekday number...

