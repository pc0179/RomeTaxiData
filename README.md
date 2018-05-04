## RomeTaxiData

# Repository Code Overview
*All code uses python3+ and postgres 9.5.12+*
*Routing engine = OSRM-backend*

Global variables such as trace start time, datum location, bounding box (8x8km) are saved in [RomeTaxiGlobalVars](RomeTaxiGlobalVars.py)

In the rome taxi trace data, one finds the following 'columns' in the .csv data file:
- taxiID
- TimeStamp
- GPS

1. Extract/Transform:
  - Timestamps to unix and datetime obj timestamps
  - reduce GPS traces to 8 sig. figs (similar to accuracy used to build OSM and google maps)
  - *"intelligent"* attributes such as day number, weekday number etc.. are added to aid lookup in PostgresDB
  - copy all to giant .csv file, before importing to Posgtgres database (currently using version 9.5.12)
  
2. Cleaning:
  - traces are quite literally *'all-over-da-place'* hence need to map-match cleverly to nearest OSM street segment
  - currently using OSRM as the back-end map-matching and routing engine

3. Questions to answer...
  - At any given time T, where are all the taxis, will need to interpolate between trace points intelligently
  - How many can communicate with each other? - LoS model....

# Background...

Multiple studies look at how vehicles move in cities and their ability to communicate with one another (V2V) and with infrastructure such as base stations (V2I).

With the advent of autonomous vehicles (AVs) looming, the current trend in wireless vehicle communication systems is the adoption of the IEEE 802.11p and/or LTE-V2X wireless standards. So far, the US, EU and Japan have agreed on a frequency band (around [5.9-6GHz](http://5gaa.org/wp-content/uploads/2017/12/5GAA-Road-safety-FINAL2017-12-05.pdf) resevered solely for communicating vehicles. 802.11p has a typical Line-of-Sight (LoS) range of around 200m and is unlikely (given it’s high frequency band) to penetrate buildings, thus making communications between vehicles on nearby but separate roads problematic.

Global vehicle ownership has only increased since 1960. [Estimates](http://www.jstor.org/stable/41323125?seq=1#page_scan_tab_contents) places the figure at around 1.2 billion vehicles; roughly split 3:1, car:buses and trucks respectively. In the US and EU there are roughly 800 and 500 vehicles per 1000 people respectively. This begs the question, in the future, will there be enough AVs such that a critical mass (most likely in cities) could communicate with each other and form a city wide AV mesh network? Could this AV mesh network be opened up to the public? It's unlikely the bit-rate will be high (envisage more ‘whatsapp’ messaging than ‘netflix’ streaming) but a free public messaging system could be useful, especially if it's decentralised, almost cost neutral  (it relies on wireless systems which will be mandatory for AVs) and doesn't impede safety critical messages between AVs. Users could communicate solely via the mesh network (i.e. sending a message from one part of the city to another via multi-hop AVs). However, to allow access to the Internet, messages will need to be routed to a base station with a fibre back-bone connection. Connected traffic light boxes (latter found at most signalised intersections) could act as semi-permanent base stations in the network, allowing both efficient scheduling AVs at junctions and to offload data transfers down their fibre connection.
Given AVs are currently prohibitively expensive; a LIDAR unit still costs 20k-ish, therefore, the inital market for AVs is most likely to be developed nations (such those comprising the EU/US) where an ageing population, high insurance and labour costs make AVs a potential solution for moving vulnerable road users a boon. AVs promises for reducing congestion might prove exagerated, afterall, if a certain amount of people want to get from A-B there's nothing you can do to elivate congestion, they will all end up sitting in traffic regards of whether a computer or human is driving. However, it does free up people's time and AVs could better platoon to save energy and driver closer to each other to increase volume of vehicles on the road. The latter claim is valid techincally but at the end of the day, 100 people on a double decker bus is still probaby the most efficient use of road space. 

In an attempt to estimate the feasibility of city wide mesh network of AVs, data from taxi traces across various cities was used. Taxi trace data is good fit for this problem since they are real trips made by real paying customers, rather than sampling census data and assuming O-D tables given residency address and work address (which by the way still only explains half of trips made by humans, the other half is seemingly random, unless Alphabet releases some data). Furthermore, given the small ratio of passengers to driver; 1:5 is probably a best case scenario and potentially a tight squeeze, whereas buses or trains tend to achieve typical ratios of 1:50 or 1:1000s respectively, AVs therefore have greatest potential at reducing costs for taxi companies. AV taxis could potentially run 24-7 stopping mostly for fuel/re-charging and for cleaning of passenger quarters, after all most new vehicles now complete 100k km before needing serious maintanence/new parts.

# Aims/Objectives...

To evalute the feasibility of an autonomous vehicular mesh network as a potential means of private, decentralised delay tolerant communication system for a city.

1. Analyse real taxi datasets where possible (see list below)
2. Evaluate effectiveness by investigating along the lines of the following params.:
    - Line-of-Sight distance between communicating taxis, how often does this happen? How does the communication capability of the mesh network of taxis vary in comparison to a less stringent communication model, such as simple 'disc' radius range
    - using LoS model, can pedestrians 'talk' to AVs? If so to what extent
    - communications system back-end? assume traffic lights and intersection have boxes wired to 'the internet' can AVs off load data/requests there?
    - Do city networks affect capability/feasibilty of system, compare for example classic radial vs grid road network structures
    
A brief (non-exhuastive list) of currently publically available taxi **trace** datasets are:
- [Rome](https://crawdad.org/roma/taxi/20140717/) 
- [San Francisco](https://crawdad.org/epfl/mobility/20090224/)
- [Shanghai](http://wirelesslab.sjtu.edu.cn/taxi_trace_data.html) (maybe, emailed twice, no reply)

Available taxi **trip** datasets:
- [NYC](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml)

# Problems to solve to achieve enternal enlightenment...

1. GPS traces are messy/noisy, they need to be filtered and map-matched to nearest road segments. It is important to take into account driving routes rather than purely matching to nearest segment, as when roads are nearby (e.g. in parallel grid structures) it could lead to false turns/deviations from original route.

2. Traces will need to *intelligently* interpolated. Since the distribution of position updates is not uniform (see [CDF update frequency plot](cdf_frequency_rome_taxi_trace_updates.pdf)) nor is it particularly 'frequent'; 90% of GPS updates are less than every 20s, median 10s approx. updates). To do this, map-matched positions of taxis will need to be interpolated along the driving segment before being divided into 1s chunks, which is likely to be the highest resolutions needed. Any further increases of resolutions are unlikely to yield bettr results whilst sacrificing computational efficiency/overall running time.

3. In the case of trip-only datasets such as NYC, entire pseudo traces are likely to be needed. To do this properly a basic traffic model might be needed, in this case, a simple work-around could be to query Alphabets Google Maps service for a subset of trips, and save the typical times and routes suggested for different days of the week, since a lot of vehicular traffic is [weekly periodic](taxis_on_duty.png) or daily if you were to divide it into 'working' week-days and weekends see [hourly taxi count figure](taxis_on_duty_by_hour.pdf).

4. Line-of-Sight model needs to take into account bends/turns in the road network as well as being bounded by buldings (if present either side of the road). Given OSM has to divide road bends into different line segments with way-points indicating start and end, in theory, if two vehicles were on the same line segment, it's unlikley they would be unable to communicate since they would be within LoS by simple virtue of having short segments in order to caputre acurately the shape of the road (however, this can quickly descend into a fractal graph mess, length of British coastlines anyone?). However, for very long segments, those >100m it might be less likely.

5. Vehicle-to-Pedestrian communication (V2P) model will need a distribution of pedestrians across the map. Census data could be used to provide an average, however, using a simple uniform distribution is a good start. 

6. It is **very** likely that there simply aren't enough taxis that took part in any of the data gathering exercises to provide meaningful analysis. Therfore, *psuedo* taxis might be used, where several days (eg all tuesday's) could be combined to provide a more *realistic* fleet of taxis/connected AVs. This would give an idea of critical mass required if such a system were to be implemented without resorting to randomly generating O-D tables/trips.




  





