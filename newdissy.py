# -*- coding: utf-8 -*-
"""
Created on Fri Aug 09 18:31:04 2019

@author: arno
"""
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import datetime

os.chdir("C:\\Users\\arno\\Downloads\\credit scoring stuff\\data\\data")

generation = pd.read_csv("generation.csv")
don_parkhill_flow = pd.read_csv("don_parkhill_flow.csv")
don_parkhill_level = pd.read_csv("don_parkhill_level.csv")
don_parkhill_level_15min= pd.read_csv("don_parkhill_level_15min.csv")

don_parkhill_flow['date'] = pd.to_datetime(don_parkhill_flow['date'])
don_parkhill_flowz = don_parkhill_flow.resample('D',on='date').mean()
don_parkhill_flowz['monthz'] = don_parkhill_flowz.index.strftime("%B")
don_parkhill_flowz['monthzZ'] = don_parkhill_flowz.index.month
don_parkhill_flowz['energy_gen'] = p2(np.log(don_parkhill_flow['flow']))
don_parkhill_flowz['energy_gen']=don_parkhill_flowz['energy_gen'].mask(cond=don_parkhill_flowz['energy_gen']>100,other=100)
don_parkhill_flowz['energy_gen']=don_parkhill_flowz['energy_gen'].mask(cond=don_parkhill_flowz['energy_gen']<=0,other=0)
don_parkhill_flowzz = don_parkhill_flowz.resample('5A').mean()
plt.plot(don_parkhill_flowzz['flow'])
plt.xlabel('Year')
plt.title('5 Year average of the flow rate',fontsize=18)
plt.ylabel('Mean flow rate')
#plt.ylabel('Standard deviation flow rate')
result = adfuller(don_parkhill_flowzz['flow'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

don_parkhill_flowz = don_parkhill_flowz.resample('M').mean()
don_parkhill_flowz.boxplot('energy_gen',by='monthzZ') 
plt.title('Barplot of average monthly power rate',fontsize=20)
plt.ylabel('Power rate')
plt.xlabel('Month')
plt.suptitle(' ')
plt.xticks(list(range(1,13)),['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],rotation=90,fontsize=10)
plt.ylim((0,100))
plt.show()
np.mean(don_parkhill_flowz['energy_gen'])*365*24

test = don_parkhill_flowz.groupby(don_parkhill_flowz.index.month).mean()
don_parkhill_flowz.boxplot('flow',by='monthzZ') 
plt.title('Barplot of average monthly flow rate',fontsize=20)
plt.ylabel('Flow')
plt.xlabel('Month')
plt.suptitle(' ')
plt.xticks(list(range(1,13)),['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],rotation=90,fontsize=10)
plt.ylim((0,70))
plt.show()
plt.plot(test)
dpfff = don_parkhill_flowz.groupby(pd.Grouper(freq='M')).mean()
don_parkhill_flowz = don_parkhill_flow.resample('3A',on='date').mean()
plt.plot(don_parkhill_flowz['flow'])
result = seasonal_decompose(don_parkhill_flowz['flow'], model='additive')
result.plot()

generation['date'] = pd.to_datetime(generation['date'],dayfirst=True)
don_parkhill_flow['date'] = pd.to_datetime(don_parkhill_flow['date'])
end_date2 = don_parkhill_flow.iloc[-1]['date']
don_parkhill_flow = don_parkhill_flow.resample('D',on='date').mean()
start_date2 = datetime(2018,10,27)
headd = generation.resample('D',on='date').mean()
headd2 = headd.loc[headd.index>=start_date2]
headd3= headd2
headd2 = headd2+0.000001
headd2['effi'] = headd2['t1_avg_power']/(9.81*headd2['t1_flow']*(headd2['us_lvl']-headd2['ds_lvl']))
headd2 = headd2.fillna(value=headd2.mean())
#efficiency in function of flow  REALLY NOICE
headd3 = headd2[headd2['effi']<1]
zz = np.polyfit(headd3['t1_flow'], headd3['effi'], 4)
p = np.poly1d(zz)
xpp = np.linspace(0, 10, 100)
plt.plot(headd3['t1_flow'],headd3['effi'], '.', xpp, p(xpp), '-')
#plt.scatter(headd3['t1_flow'],headd3['effi'])
plt.plot(flow20,efficiency20)
plt.ylim((0,1))
plt.legend(['Observation', 'fitted 4th degree polynomial','feasibility study function'])
plt.title('Efficiency in function of flow rate',fontsize=15)
plt.xlabel('Flow rate (m^3/s)')
plt.ylabel('Efficiency')


#percentiles generation flow to compare
level_puly = headd2['t1_flow'].fillna(value=headd['t1_flow'].mean()).quantile(np.array([ 0.05,0.1 ,0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]))


plt.scatter(generation['t1_flow'],generation['meter_power'],s=1)
plt.scatter(headd2['t1_flow'],headd2['meter_power'],s=10)
plt.scatter(np.log(SEPA_flow),headd2['meter_power'][:221],s=10)

zzm2 = np.polyfit(np.log(SEPA_flow)['flow'].values, headd2['meter_power'][:221], 4)
p2 = np.poly1d(zzm2)
xpp2 = np.linspace(np.exp(1.8), np.exp(4.2), 100)
plt.plot(np.log(SEPA_flow)['flow'].values,headd2['meter_power'][:221], '.', np.log(xpp2), p2(np.log(xpp2)), '-')
#plt.scatter(headd3['t1_flow'],headd3['effi'])
plt.plot(np.log(np.flip(allflow,0)),meatpower[:-1])
#plt.ylim((0,1))
plt.legend(['Observation', 'fitted 3rd degree polynomial','feasibility study function'])
plt.title('Power rate in function of flow rate',fontsize=15)
plt.xlabel('Flow rate (m^3/s)')
plt.ylim((0,105))
plt.ylabel('Power rate (kW)')

zzm = np.polyfit(headd2['t1_flow'], headd2['meter_power'], 3)
p = np.poly1d(zzm)
xpp = np.linspace(0, 10, 100)
plt.plot(headd3['t1_flow'],headd3['meter_power'], '.', xpp, p(xpp), '-')
#plt.scatter(headd3['t1_flow'],headd3['effi'])
plt.plot(flow20[:-1],meatpower[:-1])
#plt.ylim((0,1))
plt.legend(['Observation', 'fitted 3rd degree polynomial','feasibility study function'])
plt.title('Power rate in function of flow rate',fontsize=15)
plt.xlabel('Flow rate (m^3/s)')
plt.ylabel('Power rate (kW)')

plt.scatter(headd2['us_lvl']-headd2['ds_lvl'], headd2['meter_power'],s=5)
plt.scatter(generation['us_lvl']-generation['ds_lvl'],generation['meter_power'],s=1)
#model function to this for Q1
plt.scatter(headd2['t1_flow'],headd2['us_lvl']-headd2['ds_lvl'],s=10)
flowwy = headd2['t1_flow']
headdy = headd2['us_lvl']-headd2['ds_lvl']
flowwy = flowwy.fillna(value=flowwy.mean())
headdy = headdy.fillna(value=headdy.mean())

zz = np.polyfit(flowwy, headdy, 2)
p = np.poly1d(zz)
xpp = np.linspace(0, 10, 100)
plt.plot(flowwy, headdy, '.', xpp, p(xpp), '-',flow20,np.flip(oghead20,0),'-')
plt.legend(['Observation', 'fitted 2e degree polynomial','feasibility study function'])
plt.title('Head in function of flow rate',fontsize=15)
plt.xlabel('Flow rate (m^3/s)')
plt.ylabel('Head level (meter)')
#put this plot in
meatpower = np.array([0,0,10.1,20.6,23.4,33.9,37.8,51.4,57.0,70.9,74.0,79.0,87.7,96.8,100,100,100,100,100,69.8])
flow20 = np.array([0,0,0.900,1.696,1.900,2.658,2.933,3.903,4.293,5.286,5.500,5.856,6.486,7.155,7.986,9.015,10,10,10,10])
allflow = np.array([52.45,40.37,33.43,29.17,25.79,23.16,20.99,18.95,17.11,15.42,13.89,12.63,11.39,10.32,9.23,8.24,7.24,6.37,5.40])
oghead20 = np.array([1.850,1.857,1.861,1.863,1.866,1.871,1.874,1.879,1.886,1.901,1.912,1.929,1.952,1.972,1.976,1.982,1.988,2.001,2.031,2.057])
efficiency20 = np.array([0,0,0.52,0.57,0.58,0.61,0.62,0.65,0.65,0.67,0.68,0.68,0.69,0.70,0.71,0.71,0.71,0.70,0.69,0.67])
headzz = p(np.array([0,0,0.900,1.696,1.900,2.658,2.933,3.903,4.293,5.286,5.500,5.856,6.486,7.155,7.986,9.015,10,10,10,10]))
weightys = np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.01])
powerrate = np.multiply(flow20,headzz)
powerrate = np.multiply(powerrate,efficiency20)
powerrate = np.multiply(powerrate,9.81)
powerrate[powerrate>100] = 100.0
powerrate2 = np.multiply(powerrate,weightys)
yeeet = np.multiply(powerrate2,24*365/0.95)
#go from rate to total energy
totalenergy = np.sum(np.multiply(powerrate2,24*365/0.95))

#plt.ylim(0,10)
plt.show()
# and make month by month predictions

ACE_flow = headd['t1_flow']

start_date2 = datetime(2018,10,27)
end_date2 = don_parkhill_flow.index[-1]

SEPA_flow = don_parkhill_flow.loc[(don_parkhill_flow.index>=start_date2) & (don_parkhill_flow.index <= end_date2)]['flow']
np.mean(SEPA_flow)
ACE_flow = ACE_flow.loc[(ACE_flow.index >= start_date2) & (ACE_flow.index <= end_date2)]

plt.scatter(SEPA_flow,ACE_flow,s=10)
plt.title('Flow rates at Parkhill and at the turbine',fontsize=15)
plt.xlabel('Parkhill flow rate (m^3/s)')
plt.ylabel('Turbine flow rate (m^3/s)')
plt.scatter(np.log(SEPA_flow.values),ACE_flow,s=10)
plt.title('Flow rates at Parkhill and at the turbine',fontsize=15)
plt.xlabel('Logarithm of Parkhill flow rate (log(m^3/s))')
plt.ylabel('Turbine flow rate (m^3/s)')
loggy=np.log(SEPA_flow.values)
SEPA_flow.corr(ACE_flow)

dp_flow_M = don_parkhill_flow.resample('A',on='date').mean()
plt.plot(dp_flow_M.index,dp_flow_M['flow'])
dp_dates = don_parkhill_flow.index

np.mean(dp_flow_M['flow'])
ACE_flow = ACE_flow.fillna(value=ACE_flow.mean())
z = np.polyfit(loggy, ACE_flow.values, 1)
p = np.poly1d(z)
xp = np.linspace(np.exp(1.5), 60, 100)
plt.plot(loggy, ACE_flow, '.', np.log(xp), p(np.log(xp)), '-')
plt.plot(np.log(np.flip(allflow,0)),flow20[:-1])
plt.title('Linear fit for flow rates at Parkhill and at the turbine',fontsize=15)
plt.xlabel('Logarithm of Parkhill flow rate (log(m^3/s))')
plt.ylabel('Turbine flow rate (m^3/s)')
plt.legend(['Observation', 'fitted linear function','SEPA flow restrictions'])
plt.ylim(0,10)
plt.show()

SEPA_flowl = np.log(SEPA_flow)
SEPA_flowl.corr(ACE_flow)

alltflows = p(np.log(don_parkhill_flow['flow']))
alltflowsS = pd.Series(alltflows)
alltflowsS=alltflowsS.mask(cond=alltflows>10,other=10)
alltflowsS=alltflowsS.mask(cond=alltflows<0,other=0)

alltflowsS = pd.Series(data=alltflowsS.values, index=dp_dates)

plt.plot(alltflowsS)
np.mean(alltflowsS)
np.mean(ACE_flow)
np.sum(alltflowsS.values<0.00001)/float(len(alltflowsS.values))

alltflowsYE = alltflowsS.resample('A').mean()
plt.plot(alltflowsYE)
np.mean(alltflowsYE)

##############################
## Finding long term trends ##
##############################
# this to look at YEARLY FLOW TREND
# LEVEL TREND
# relationship flow and level and from this get a more long term level trend
# use generation level and link this to flow 

#upstream level
don_parkhill_level_15min['timestamp'] = pd.to_datetime(don_parkhill_level_15min['timestamp'])
dp_lev_M = don_parkhill_level_15min.resample('M',on='timestamp').mean()
dp_lev_D = don_parkhill_level_15min.resample('D',on='timestamp').mean()
plt.plot(dp_lev_M)
#generation level
gen_lev = headd['us_lvl']-headd['ds_lvl']
gen_flow = headd['t1_flow']
gen_lev_M = gen_lev.resample('M').mean()
gen_lev_D = gen_lev.resample('D').mean()
gen_flow_D = gen_flow.resample('D').mean()
plt.plot(gen_lev_M)
plt.plot(dp_lev_M)
plt.show()
#flow to match both intervals
interval_dp = (dp_lev_D.index[0],dp_lev_D.index[-1])
interval_gen = (gen_lev_D.index[0],gen_lev_D.index[-1])
interval_flow = (don_parkhill_flow.index[0],don_parkhill_flow.index[-1])

dp_range = don_parkhill_flow.loc[(don_parkhill_flow.index >= interval_dp[0]) & (don_parkhill_flow.index <= interval_flow[1])]
dp_lev_D2 = dp_lev_D.loc[(dp_lev_D.index >= interval_dp[0]) & (dp_lev_D.index <= interval_flow[1])]
dp_lev_D3 = dp_lev_D2.loc[(dp_lev_D2.index >= interval_gen[0]) & (dp_lev_D2.index <= interval_flow[1])]

dp_gen = don_parkhill_flow.loc[(don_parkhill_flow.index >= interval_gen[0]) & (don_parkhill_flow.index <= interval_flow[1])]
gen_D2 = gen_lev_D.loc[(gen_lev_D.index >= interval_gen[0]) & (gen_lev_D.index <= interval_flow[1])]
gen_flow_D2 = gen_flow_D.loc[(gen_lev_D.index >= interval_gen[0]) & (gen_flow_D.index <= interval_flow[1])]

plt.scatter(dp_range,dp_lev_D2,s=0.8)
plt.scatter(dp_gen['flow'],gen_D2)
plt.scatter(dp_lev_D3,gen_D2)
plt.scatter(dp_lev_D3,gen_flow_D2)
plt.scatter(dp_gen,gen_flow_D2)
