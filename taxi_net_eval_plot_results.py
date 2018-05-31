#pc0179 on the Kboard
# quick mini script to plot some results from rome taxi data

import pandas as pd
import matplotlib.pyplot as plt

taxi_net_results = pd.read_csv('taxi_net_eval_combined_day3.csv')

plt.figure()
plt.plot(taxi_net_results.time, taxi_net_results.total_conns,'-ob', label = '100m disc range')
plt.plot(taxi_net_results.time,taxi_net_results.los,'-*k', label = 'Within LoS')
plt.ylabel('# Taxis potentially able to communicate')
plt.xlabel('time/s')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(taxi_net_results.time,taxi_net_results.total_conns.cumsum(),'-b', label = 'Combined(noLoS+LoS), within 100m disc range')
plt.plot(taxi_net_results.time,taxi_net_results.los.cumsum(),'-g', label = 'Within Line-of-Sight')
plt.plot(taxi_net_results.time,taxi_net_results.nolos.cumsum(),'-r', label='No Line-of-Sight')
plt.xlabel('time/s')
plt.ylabel('cumsum of potential V2V connections')
plt.legend(loc='upper left')
plt.show()
