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

With the advent of autonomous vehicles (AVs) looming, wirelessly connecting AVs such that they are able to communicate with one another (V2V) and with infrastructure (V2I) has been the subject of much research. Very briefly, physical experiments measuring V2V performance in [harsh environments](https://ieeexplore.ieee.org/document/6799812/) as well ability to stream [video](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6649286)and make [VOIP calls](http://webhost.services.iit.cnr.it/staff/paolo.santi/papers/IEEEVTC12.pdf) have been conducted for vehicles. All papers highlight potential limitations and the technical capabilities of two standards likely to be adopted for wireless communication systems between vehicles; IEEE 802.11p (also known as DSRC) and LTE-V2X. Both have received much [research](https://link.springer.com/content/pdf/10.1007%2Fs41650-017-0022-x.pdf) attention, however, the automotive industry has yet to produce vehicles with any such technology, hence a single dominating global standard has not emerged yet. Nonetheless, the US, EU and Japan have agreed on a frequency band (around [5.9-6GHz](http://5gaa.org/wp-content/uploads/2017/12/5GAA-Road-safety-FINAL2017-12-05.pdf) resevered solely for communicating vehicles. IEEE 802.11p and LTE-V2X have theoretical Line-of-Sight (LoS) range of up to 1000m, however, real world experiments show a rapid decrease in bit-rate at distances over over [250m](https://pdfs.semanticscholar.org/e995/3f93d1eaafa3025a4e87f25713d89fca491f.pdf)                   between vehicles. In harsher environments, such those found in open quarries where dust clouds are typical, wireless ranges decrease further to around [100-150m](https://ieeexplore.ieee.org/document/6799812/). Furthermore, given the high (5.9GHz) frequency band, it is highly unlikely that either standard is able to penetrate buildings, thus making communications between vehicles on nearby but separate roads problematic

Global vehicle ownership has only increased since 1960. [Estimates](http://www.jstor.org/stable/41323125?seq=1#page_scan_tab_contents) places the figure at around 1.2 billion vehicles; roughly split 3:1, car:buses and trucks respectively. In the US and EU there are roughly 800 and 500 vehicles per 1000 people respectively. This begs the question, in the future, will there be enough AVs such that a critical mass (most likely in cities) could communicate with each other and form a city wide AV mesh network? Could this AV mesh network be opened up to the public? It's unlikely the mean user experienced bit-rate will be high (envisage more ‘whatsapp’ messaging than ‘netflix’ streaming) given the complexity of enacting fast handovers as well as routing packets through a moving (i.e. topologically changing) mesh network, still, a potentially ‘free’ public messaging system could be useful. Even better, if the system is decentralised and almost cost neutral (since, in theory, it relies on wireless systems which will be mandatory for AVs) and doesn't impede safety critical messages between AVs; users could communicate solely via the mesh network (i.e. sending a message from one part of the city to another via multi-hop AVs). However, to allow access to the Internet, packets will need to be routed to a base station with a fibre back-bone connection. Connected traffic light boxes (found at most signalised intersections) could act as semi-permanent base stations in the AV mesh network, thus allowing both efficient scheduling AVs at junctions and offloading of data transfers through their fibre connection. Vehicles themselves could in theory be equipped with both short range and long range (i.e. sub GHz) wireless systems and act as relays to a backbone 3-4G cellular base stations, already prevalent in many cities across the globe.

Given AVs are currently prohibitively expensive; a single LIDAR unit still costs almost as much as a single typical hatchback, therefore, the initial market for AVs is most likely to be developed nations (such those comprising the EU/US) where an ageing population, high insurance and labour costs make AVs a potential solution for moving vulnerable (or wealthy, non-driving) road users. However, AVs promises for reducing congestion might prove exaggerated. After all, if a large number of people want to get from A-B (imagine typical rush-hour in most cities) vehicles will all end up sitting in traffic regardless of whether a computer or human is driving, since there is a physical limit as to how many vehicles can ‘flow’ through a road network. Nonetheless, AVs do have the potential to free up people's time, remove the need to learn to drive, in theory, could sel-organise into motorway platoons to save energy whilst also being able to drive (assuming computer reaction times are much faster than human driver’s) closer to each other to increase volume of vehicles on the road. The latter claim is valid technically, however, given there will be for many years, lingering human driven vehicles, on average it’s unlikely to increase volume flow by much. Note, that at least in terms of city transport, few systems beat 100+ people on a double decker bus in terms of humans being transported per road area utilised.

With regards to capabilities and potential applications of wide spread V2V comms. Research has mainly focused on improving intersection efficiency and safety through clever scheduling depending on traffic conditions at an intersection (collections of various papers: [mit-portugal] (https://www.mitportugal.org/education/transportation/research/intelligent-transportation-systems), [VTL](http://www.ibr.cs.tu-bs.de/courses/ws1011/advnet1/paper/vanet2010/p85.pdf) and [more VTL](https://users.ece.cmu.edu/~tonguz/vtl/publications.html)). To evaluate and estimate the feasibility of such V2V networks, [researchers](https://ieeexplore.ieee.org/document/7895148/) (and [here]((https://ieeexplore.ieee.org/abstract/document/7511319/)) have often used taxi trace data as their mobility model for the communicating nodes (i.e. vehicles). Taxi trace data has certain key features;
They are real trips made by paying customers, who in the future could use an AV taxi
Data is often provided with resolution of 10-30 seconds allowing for ‘reasonable’ estimates based on relative speeds, locations etc… of communicating nodes
Reduces number of assumptions and external data needed to generate trips from other data sources such as census data, travel surveys etc…
Data is free and publicly available (see links below.) allowing for efficient comparisons of various simulations and estimates of network performance/topology etc...

Furthermore, given the small ratio of driver:passengers in a taxi; 1:5 is probably a best case scenario (and potentially a tight squeeze depending mean group member size/weight), taxi companies stand most to gain in terms of savings made by having a computer to drive the vehicle. On the other hand, buses and trains tend to achieve much higher driver:passenger ratios, typically of 1:50-100 or 1:1000’s respectively, which means automation will produce less savings (as you are replacing far fewer drivers). Taking this further, AV taxis could potentially run 24-7 stopping only for fuel/re-charging, maintenance and for cleaning of passenger quarters.

# Aims/Objectives...

To evaluate the feasibility of an autonomous vehicular mesh network as a potential means of private, decentralised delay tolerant communication system for a city.

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

# Problems to solve to achieve eternal enlightenment...

1. GPS traces are messy/noisy, they need to be filtered and map-matched to nearest road segments. It is important to take into account driving routes rather than purely matching to nearest segment, as when roads are nearby (e.g. in parallel grid structures) it could lead to false turns/deviations from original route.

2. Traces will need to *intelligently* interpolated. Since the distribution of position updates is not uniform (see [CDF update frequency plot](cdf_frequency_rome_taxi_trace_updates.pdf)) nor is it particularly 'frequent'; 90% of GPS updates are less than every 20s, median 10s approx. updates). To do this, map-matched positions of taxis will need to be interpolated along the driving segment before being divided into 1s chunks, which is likely to be the highest resolutions needed. Any further increases of resolutions are unlikely to yield better results whilst sacrificing computational efficiency/overall running time.

3. In the case of trip-only datasets such as NYC, entire pseudo traces are likely to be needed. To do this properly a basic traffic model might be needed, in this case, a simple work-around could be to query Alphabets Google Maps service for a subset of trips, and save the typical times and routes suggested for different days of the week, since a lot of vehicular traffic is [weekly periodic](taxis_on_duty.png) or daily if you were to divide it into 'working' week-days and weekends see [hourly taxi count figure](taxis_on_duty_by_hour.pdf).

4. Line-of-Sight model needs to take into account bends/turns in the road network as well as being bounded by buildings (if present either side of the road). Given OSM has to divide road bends into different line segments with way-points indicating start and end, in theory, if two vehicles were on the same line segment, it's unlikely they would be unable to communicate since they would be within LoS by simple virtue of having short segments in order to capture accurately the shape of the road (however, this can quickly descend into a fractal graph mess, length of British coastline anyone?). However, for very long segments, those >100m it might be less likely.

5. Vehicle-to-Pedestrian communication (V2P) model will need a distribution of pedestrians across the map. Census data could be used to provide an average, however, using a simple uniform distribution is a good start. 

6. It is **very** likely that there simply aren't enough taxis that took part in any of the data gathering exercises to provide meaningful analysis. Therefore, *psuedo* taxis might be used, where several days (eg all tuesday's) could be combined to provide a more *realistic* fleet of taxis/connected AVs. This would give an idea of critical mass required if such a system were to be implemented without resorting to randomly generating O-D tables/trips.




  





