#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd

psql_output_file = 'psql_query_output.csv'
psql_cols=['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','Lat1','Long1','x','y']

df = pd.read_csv(psql_output_file, names = psql_cols)

