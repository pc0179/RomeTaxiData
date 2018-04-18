import matplotlib.pyplot as plt
import pandas as pd
import RomeTaxiGlobalVars as RTGV
import datetime as dt


taxi_hr_count = pd.read_csv('/home/pdawg/RomeTaxiData/pd_taxi_count_per_hour.csv')

taxi_hr_count['sim_t'] = taxi_hr_count['sim_t']/3600


#using datetime instead
taxi_hr_count['sim_t'] = pd.date_range(RTGV.sim_start_time,periods=24*27, freq='H')


#plotting figure
fig, ax = plt.subplots()
plt.plot(taxi_hr_count['sim_t'],taxi_hr_count['num_taxis'],'-*')
plt.xlabel('Time/[Hrs]')
plt.ylabel('Number of taxis on duty')
ax.grid(color='k', linestyle='--', linewidth=0.5)
plt.show()
#plt.savefig('/home/pdawg/RomeTaxiData/Num_taxis_hourly.pdf')




