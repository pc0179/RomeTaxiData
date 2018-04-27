## RomeTaxiData
# Rome Taxi Data-set February 2014


# Quick Intro.

Multiple studies look at how vehicles move in cities and their ability to communicate with one another (V2V) and with infrastructure such as base stations (V2I).

With the advent of autonomous vehicles (AVs) looming, the current trend in wireless vehicle communication systems is the adoption of the IEEE 802.11p wireless standard. So far, the US and EU have agreed on a frequency band (around 5.9GHz) resevered solely for communicating vehicles. Even thought Wi-Fi has typical Line-of-Sight (LoS) ranges 50-200m it is unlikely to penetrate buildings, thus making communications between vehicles on nearby but different roads unlikely. Furhermore, this introduces a 'time cost' or penalty at un-controlled (i.e. those without a traffic light enforcement system) junctions where AVs will have to resort to crawling forward, like human drivers, checking at every small increment.

Global vehicle ownership has only increased since 1960. Estimates made by Wards???? put the current figure at around 1.2 billion; roughly split 3:1, car:buses and trucks respectively. In the US and EU there are roughly 800 and 500 vehicles per 1000 people respectively. This begs the question, are there enough such that a critical masss of AVs (most likely in cities) could 'talk' to each other and form a city wide mesh network? Could this network be opened up to the masses, it's unlikely the bit-rate will be large (more whatsapp messaging than netflix streaming) but a free public messaging system could be useful, especially if it's decentralised and doesn't impede safety critical messages between AVs. Furthermore, parked vehicles could act as semi-permentant nodes in the network, skillfully scheduling AVs around blind junctions and or acting as access-points for the public (think of them as relay nodes) into the AV mesh network.  

Given AVs are currently prohibitively expensive; a LIDAR unit still costs 20k-ish, although this will likely decrease, AVs are likely to require multiple sensors to have suitable redudancy (just ask UBER, whose vehilce fitted with just one lidar unit killed a pedestrian, similarly Tesla inisists on having just one radar unit, their autopilot feature is responsible for two deaths so far...). Therefore, the inital market for AVs is most likely to be developed nations (such those comprising the EU/US) where an ageing population, high insurance and labour costs make AVs a potential solution for moving vulnerable road users a boon. AVs promises for reducing congestion might prove exagerated, afterall, if a certain amount of people want to get from A-B there's nothing you can do to elivate congestion, they will all end up sitting in traffic regards of whether a computer or human is driving. However, it does free up people's time and AVs could better platoon to save energy and driver closer to each other to increase volume of vehicles on the road. The latter claim is valid techincally but at the end of the day, 100 people on a double decker bus is still probaby the most efficient use of road space. 

In an attempt to estimate the feasibility of city wide mesh network of AVs, data from taxi traces across various cities is used. Taxi trace data is good fit for this problem since they are real trips made by paying customers, rather than sampling census data and assuming O-D tables given residency address and work address (which by the way still only explains half of trips made by humans, the other half is seemingly random, unless Alphabet releases some data). Furthermore, given the small ratio of passengers to driver; 1:5 is probably a best case scenario and potentially a tight squeeze, whereas buses or trains tend to achieve typical ratios of 1:50 or 1:1000s respectively, AVs therefore have greatest potential at reducing costs for taxi companies. AV taxis could potentially run 24-7 stopping mostly for fuel/re-charging and for cleaning of passenger quarters, after all most new vehicles now complete 100k km before needing serious maintanence/new parts.

Currently publically available taxi **trace** datasets for taxis are:
- Rome 
- San Francisco
- Shenzen (maybe, emailed twice, no reply)

Available taxi **trip** datasets:
- NYC





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

