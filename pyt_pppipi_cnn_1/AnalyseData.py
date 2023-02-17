import numpy as np
import cupy as cp
import pandas as pd
import datetime
import os
from numpy.lib import recfunctions as rfn
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#########################################################
# Definitions for importing csv files.
digiParNames=['run','event','trackId','gid','layer','cell','time','charge','turnid']
digiParDtypes={'run':np.int32,'event':np.uint32,'trackId':np.int16,'gid':np.uint16,\
               'layer':np.uint16,'cell':np.uint16,'time':np.float32,'charge':np.float32, \
               'turnid':np.uint16}

mcParNames=['run','event','trackId','pdg','px','py','pz','x','y','z','time','turn']
mcParDtypes={'run':np.int32,'event':np.uint32,'trackId':np.int16,'pdg':np.int16,\
             'px':np.float32,'py':np.float32,'pz':np.float32,'x':np.float32,'y':np.float32,\
             'z':np.float32,'time':np.float32,'turn':np.uint16}

# Importing the csv file of the 20000 events of j-psi decay to pppipi.
digimu = pd.DataFrame()
digimuwithback = pd.DataFrame()
digimut0withback = pd.DataFrame()

mcmu = pd.DataFrame()
mcmuwithback = pd.DataFrame()
mcmut0withback = pd.DataFrame()

digi_pppipi_20000_nonoise = pd.read_csv('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/generate/digi_pppipi_5_0_nonoise.csv',header=0,sep=',',names=digiParNames,dtype=digiParDtypes)
mc_pppipi_20000_nonoise = pd.read_csv('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/generate/mc_pppipi_5_0_nonoise.csv',header=0,sep=',',names=mcParNames,dtype=mcParDtypes)

digi_pppipi_20000 = pd.read_csv('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/generate/digi_pppipi_5_0.csv',header=0,sep=',',names=digiParNames,dtype=digiParDtypes)
mc_pppipi_20000 = pd.read_csv('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/generate/mc_pppipi_5_0.csv',header=0,sep=',',names=mcParNames,dtype=mcParDtypes)

#########################################################
# Wires at each layer
import math
wires = np.array([40, 44, 48, 56, 64, 72, 80, 80, 76, 76, 88, 88, 100, 100, 112, 112, 128, 128, 140, 140, \
                 160, 160, 160, 160, 176, 176, 176, 176, 208, 208, 208, 208, 240, 240, 240, 240, \
                  256, 256, 256, 256, 288, 288, 288])

# gid and angle based on cell and laayer
def gidf(layer, cell):
    return wires[0:layer].sum() + cell

def anglef(layer, cell):
    return ((360 * cell) / (wires[layer] * 9))

# The diameter of the wires are about 12 mm. First layer is in 71mm radius of the center.
# 1st tube radius: about 70mm to 170mm, 1st gap: 167mm to 191mm
# 2nd tube radius: about 170mm to 380mm, 2nd gap: 379mm to 393mm
# 3rd and 4th gap: about 12mm
# radius of the last layer: about 760mm

#########################################################
# sort
#dfnr = digi_pppipi_20000.sort_values('time').drop_duplicates(subset = ['event', 'gid'], keep = 'first')
dfnr = digi_pppipi_20000.sort_values(by=['time', 'event', 'trackId', 'gid']).drop(['run', 'turnid'], axis=1)
dfnr['hitid'] = np.where((dfnr.trackId < 0), False, True)
dfnr['angleid'] = anglef(dfnr['layer'], dfnr['cell']).astype(np.uint16)
dfnr['angle'] = (6.283 / wires[dfnr['layer']]) * dfnr['cell']

#dfnr_nonoise = digi_pppipi_20000_nonoise.sort_values('time').drop_duplicates(subset = ['event', 'gid'], keep = 'first')
dfnr_nonoise = digi_pppipi_20000_nonoise.sort_values(by=['time', 'event', 'trackId', 'gid']).drop(['run', 'turnid'], axis=1)
dfnr_nonoise['angle'] = (6.283 / wires[dfnr_nonoise['layer']]) * dfnr_nonoise['cell']

# Let's see if there is any repeated hits (There is none!)
dfnr = dfnr.sort_values('time').drop_duplicates(subset = ['event', 'gid'], keep = 'first')
dfnr = dfnr.sort_values(by=['event', 'gid'])

print('digi_pppipi_20000', dfnr, '\ndigi_pppipi_20000_nonoise', dfnr_nonoise)
print('dfnr after deleting the probable repeated events', dfnr)

#########################################################
# make array and structured array
dfarray = np.array(dfnr.to_numpy(), dtype=np.float32)

dty = np.dtype([('event', np.uint64), ('trackId', np.int16), ('gid', np.uint64), ('layer', np.uint64), \
                  ('cell', np.uint64), ('time', np.float32), \
                  ('charge', np.float32),  ('hitid', bool), \
                  ('angleid', np.uint16), ('angle', np.uint16)])
dfstrarray = rfn.unstructured_to_structured(dfarray, dty)

print('\ndata frame as an array', dfarray, '\nstructured array', dfstrarray)

#########################################################
# TrackIds after background mixing, their frequency, and the average of features.
df = dfnr.drop('event', axis=1)
dfmean = df.groupby('trackId').mean()
dfmean['frequency'] = df.groupby('trackId').size()
print('\n\naverage values when grouped by trackId', dfmean)

# TrackIds after background mixing, their frequency, and the average of properties.
df_nonoise = dfnr_nonoise.drop('event', axis=1)
dfmean_nonoise = dfnr_nonoise.groupby('trackId').mean()
dfmean_nonoise['frequency'] = df_nonoise.groupby('trackId').size()
print('\n\naverage values when grouped by trackId for the dataset that has no noise (eventmixer is not used)', dfmean_nonoise)

# Positive trackIds (1000:1003) are for replaced hits and negative ones (-996:-1000) are added hits.
# To train the neural network, we consider (1000) as signal hits since they are exactly 
# the same hits, just with a different id.
# Negative indexes other than -1000 and -999 are also negligible.
# Hence, the followings are signal and background hit sets. Other trackIds are negligible.
dfbac = dfnr[(dfnr.trackId < 0)]# | (dfnr_low.trackId == 1010) | (dfnr_low.trackId == 1011)]
dfsig = dfnr[(dfnr.trackId >= 0)]

dfbacalotnoise = dfnr[(dfnr.trackId < 0) | (dfnr.trackId >= 1000)]
dfsigalotnoise = dfnr[(dfnr.trackId >= 0)]

dfsig['angle'] = (6.283 / wires[dfsig['layer']]) * dfsig['cell']
dfbac['angle'] = (6.283 / wires[dfbac['layer']]) * dfbac['cell']

dfsigalotnoise['angle'] = (6.283 / wires[dfsigalotnoise['layer']]) * dfsigalotnoise['cell']
dfbacalotnoise['angle'] = (6.283 / wires[dfbacalotnoise['layer']]) * dfbacalotnoise['cell']

dfnr_nonoise['angle'] = (6.283 / wires[dfnr_nonoise['layer']]) * dfnr_nonoise['cell']

print('\nnoise dataframe:', dfbac)
print('\nsignal dataframe:', dfsig)

#########################################################
print('\nmc_pppipi_20000', mc_pppipi_20000)
print('\nmc_pppipi_20000_nonoise', mc_pppipi_20000_nonoise)
print('\n 2nd event in mc_pppipi_20000', mc_pppipi_20000[mc_pppipi_20000.event == 2])
print('\n 2nd event in mc_pppipi_20000_nonoise', mc_pppipi_20000_nonoise[mc_pppipi_20000_nonoise.event == 2])
print('\nevent 1260 as another example', mc_pppipi_20000.loc[mc_pppipi_20000.event == 1260])
#dfnrmc = mc_pppipi_20000.copy()
dfmc = mc_pppipi_20000.drop(['event', 'run'], axis=1)
dfmeanmc = dfmc.groupby('pdg').mean()
dfmeanmc['frequency'] = dfmc.groupby('pdg').size()
print('\naverage values when grouped by pgd', dfmeanmc)

########################## plots ########################

#########################################################
# Number of signal hits for each event.
dfnr_2 = dfnr[dfnr['hitid'] == True]
dfnr_3 = dfnr_2.groupby('event').count()['hitid']
truestat = dfnr_3.to_numpy()

# if there is no signal hits in an event, we 
for i in range(20000):
    if (dfnr_2.loc[dfnr_2.event==i]['time'] > 0).sum()==0:
        print('\nthere is no signal hit in event number', i)

# Total number of hits for each event
dfhitstat = pd.DataFrame(dfnr.groupby('event').size())
dfhitstat = dfhitstat.drop(dfhitstat[0][12278]).drop(dfhitstat[0][15965]).to_numpy()

# Total number of hits up to any event
ste = np.zeros(shape=(19998), dtype=np.uint64)
ste[0] = dfhitstat[0]
for i in range(1, 19998):
    ste[i] = ste[i-1] + dfhitstat[i]

print('\ntotal number of hits of each event', dfhitstat, '\nshape', dfhitstat.shape, '\naverage', dfhitstat.mean(), '\nsum', dfhitstat.sum())
print('\narray of the number of signal hits of each event', truestat, '\nshape', truestat.shape, '\naverage', truestat.mean())
print('\nsum up to any event', ste)

#########################################################
# histograms of the data set
plt.subplots(figsize=(12, 18))
plt.subplot(5, 2, 1)
plt.title('rawtime distribution of all hits')
plt.hist(mc_pppipi_20000['time'],\
         bins = 100, range=[-20, 2000])

plt.subplot(5, 2, 2)
plt.title('events t0')
plt.hist(mc_pppipi_20000['time'],\
         bins = 20, range=[640, 680])

plt.subplot(5, 2, 3)
plt.title('trackId')
plt.hist(mc_pppipi_20000['trackId'],\
         bins = 50, range=[-1100, 1100])

plt.subplot(5, 2, 4)
plt.title('trackId')
plt.hist(mc_pppipi_20000['trackId'],\
         bins = 50, range=[-1, 10])

plt.subplot(5, 2, 5)
plt.title('charge')
plt.hist(digi_pppipi_20000['charge'],\
         bins = 50, range=[-1100, 1100])

plt.subplot(537)
plt.title('px')
plt.hist(mc_pppipi_20000['px'],\
         bins = 50, range=[-3, 3])

plt.subplot(538)
plt.title('py')
plt.hist(mc_pppipi_20000['py'],\
         bins = 50, range=[-3, 3])

plt.subplot(539)
plt.title('pz')
plt.hist(mc_pppipi_20000['pz'],\
         bins = 50, range=[-3, 3])

plt.subplot(5, 3, 10)
plt.title('x')
plt.hist(mc_pppipi_20000['x'],\
         bins = 50, range=[0, 0.5])

plt.subplot(5, 3, 11)
plt.title('y')
plt.hist(mc_pppipi_20000['y'],\
         bins = 50, range=[-0.1, 0])

plt.subplot(5, 3, 12)
plt.title('z')
plt.hist(mc_pppipi_20000['z'],\
         bins = 50, range=[-3, 3])

plt.subplot(5, 3, 13)
plt.title('pdg')
plt.hist(mc_pppipi_20000['pdg'],\
         bins = 30, range=[-15, 15])

plt.subplot(5, 3, 14)
plt.title('trackId')
plt.hist(mc_pppipi_20000['trackId'], bins=10, range=[0, 10])

mc_pppipi_20000['p'] = (mc_pppipi_20000['px']**2 + mc_pppipi_20000['px']**2 + mc_pppipi_20000['px']**2)**0.5
plt.subplot(5, 3, 15)
plt.title('p')
plt.hist(mc_pppipi_20000['p'], bins = 50, range=[-3, 3])

t = datetime.datetime.now()
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_1/dataanalysis/{t}\
  Histogram_of_the_features.png', bbox_inches='tight')

#########################################################
# histogram of the number of hits
plt.figure(figsize=(12,8))
plt.hist(dfsig.groupby('event').size(),\
         bins = 70, range=[0, 500], alpha=0.7, \
         label = ['signal'])
plt.hist(dfbac.groupby('event').size(),\
         bins = 70, range=[0, 500], alpha=0.7, \
         label = ['background'])
plt.legend()
plt.xlabel('number of hits', fontsize=16)
plt.title('histogram of the number of hits in the signal and background', fontsize=18)
plt.tick_params(length=6, width=2, labelsize=12)
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_1/dataanalysis/{t}\
  Histogram_of_the_number_of_hits.png', bbox_inches='tight')

# Average of features for noise and signal
print(dfbac.drop(['event', 'trackId'], axis=1).mean(), dfsig.drop(['event', 'trackId'], axis=1).mean())

# Database of all the hits which have more noise than signal hits
dfsigbac = pd.DataFrame() 
dfsigbac['signum'] = dfsig.groupby(['event']).size()
dfsigbac['bacnum'] = dfbac.groupby(['event']).size()
dfsigbac['highnoise'] = dfsigbac['signum'] < dfsigbac['bacnum']
dfsigbac['bacnum'] = dfsigbac['bacnum'].fillna(0).astype(int)
dfhighnoise = dfsigbac.loc[(dfsigbac.highnoise == 1)].reset_index()
print('\n hits wich have more noise that signal hits:', dfsigbac['highnoise'].sum(), dfsigbac)

#########################################################
# time average for hits of each layer
meantimes = np.zeros(43)
meantimeb = np.zeros(43)
for i in range(43):
    meantimes[i] = dfsig.loc[dfsig.layer==i]['time'].mean()
    meantimeb[i] = dfbac.loc[dfbac.layer==i]['time'].mean()
plt.figure(figsize=(9,7))
plt.plot(range(43), meantimes, '.', label='signal')
plt.plot(range(43), meantimeb, '.', label='background')
plt.xlabel('layer', fontsize=14)
plt.ylabel('time average', fontsize=14)
plt.title('time average of the hits of each layer', fontsize=14)
plt.legend()
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_1/dataanalysis/{t}\
  Time average of the hits of each layer.png', bbox_inches='tight')

#########################################################
# Historgram of the features
plt.subplots(figsize=(10, 10))
plt.subplot(3, 2, 1)
plt.hist([dfsig['time'], dfbac['time']],\
         bins = 50, range=[-200, 1500],\
         label = ['signal rawtime distribution', 'background rawtime distribution'], density=True)
plt.legend()

plt.subplot(3, 2, 2)
plt.hist([dfsig['charge'], dfbac['charge']], bins = 50, range=[200, 1200],\
         label = ['signal charge distribution', 'background hits have charge=0'])
plt.legend()

plt.subplot(3, 1, 2)
plt.hist([dfsig['gid'], dfbac['gid']],\
         bins = 100, range=[0, 6800],\
         label = ['signal gid distribution', 'background gid distribution'])
plt.legend()

plt.subplot(3, 2, 5)
plt.hist([dfsig['cell'], dfbac['cell']],\
         bins = 50, range=[0, 290],\
         label = ['signal cell distribution', 'background cell distribution'])
plt.legend()

plt.subplot(3, 2, 6)
plt.hist([dfsig['layer'], dfbac['layer']],\
         bins = 50, range=[0, 45],\
         label = ['signal layer distribution', 'background layer distribution'])
plt.legend()
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_1/dataanalysis/{t}\
  Histogram_of_the_features.png', bbox_inches='tight')

#########################################################
# Histogram of the features.
import matplotlib.colors as mcolors
from matplotlib import colors
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(14, 5)) #create a canvas, tell matplotlib it's 3"d
ax = fig.add_subplot(121)#, projection='3d')
plt.hist2d(dfsig['time'].to_numpy(), dfsig['gid'].to_numpy(), \
                                      bins=(140, 700), range=[[0, 1500], [0, 7000]], \
                                      norm = colors.Normalize(0, 100))#, cmap=plt.cm.YlGnBu)#viridis)
plt.colorbar()
plt.xlabel('time', fontsize=16)
plt.ylabel('gid', fontsize=16)
plt.tick_params(length=6, width=2, labelsize=12)
plt.title('histogram of the signal hits', fontsize=18)

ax2 = fig.add_subplot(122)#, projection='3d')
plt.hist2d(dfbac['time'].to_numpy(), dfbac['gid'].to_numpy(), \
                                      bins=(140, 700), range=[[0, 1500], [0, 7000]], \
                                      norm = colors.Normalize(0, 100))#, norm=colors.LogNorm())
plt.colorbar()
plt.xlabel('time', fontsize=16)
plt.ylabel('gid', fontsize=16)
plt.tick_params(length=6, width=2, labelsize=12)
plt.title('histogram of the background hits', fontsize=18)
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_1/dataanalysis/{t}\
  Histogram_of_the_features2.png', bbox_inches='tight')

#########################################################
# animation of the feature space of a group of events
'''from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

evf = 599
evt = 600                     #all the hits of events from evf to evt are ploted.
hns1 = truestat[0:evf].sum()  
hnb1 = dfhitstat[0:evf].sum() - truestat[0:evf].sum()
hns2 = truestat[0:evt].sum()  
hnb2 = dfhitstat[0:evt].sum() - truestat[0:evt].sum()

xs = (7 + dfsig['layer'][hns1:hns2]) * np.cos(dfsig['angle'][hns1:hns2])
ys = (7 + dfsig['layer'][hns1:hns2]) * np.sin(dfsig['angle'][hns1:hns2])
zs = dfsig['time'][hns1:hns2]

xb = (7 + dfbac['layer'][hnb1:hnb2]) * np.cos(dfbac['angle'][hnb1:hnb2])
yb = (7 + dfbac['layer'][hnb1:hnb2]) * np.sin(dfbac['angle'][hnb1:hnb2])
zb = dfbac['time'][hnb1:hnb2]

fig = plt.figure(figsize=(12, 12))
ax = plt.gca(projection='3d')
plt.title(F'XY position vs time for hits of {evt-evf} events\n {hns2-hns1} signal hits and {hnb2 - hnb1} background hits')
#ax.grid(False)

def init():
    ax.scatter(xs, ys, zs, s=8, label='signal hits')
    ax.scatter(xb, yb, zb, s=8, label='background hits')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('time')
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.legend()
    return fig

def animate(i):
    ax.view_init(-180 * int(i / 180) + (i % 180), i)
    return fig

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=15, blit=True)
writervideo = animation.FFMpegWriter(anim, fps=60) 
anim.save(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_1/dataanalysis/{t}\
  Animation_of_the_features.mp4', writer=writervideo)'''

#########################################################
# polar plot and cell-layer plot of sample events
evnum = 410
fig = plt.figure(figsize=(24, 16)) 
fig.add_subplot(321, projection='polar')
plt.plot(dfsig.loc[dfsig.event == evnum]['angle'], 5 + dfsig.loc[dfsig.event == evnum]['layer'], \
          '.', label = 'signal', markersize=8)
plt.plot(dfbac.loc[dfbac.event == evnum]['angle'], 5 + dfbac.loc[dfbac.event == evnum]['layer'], \
          '.', label = 'background', markersize=8, color='red')
signum1 = dfsigbac['signum'][evnum]
bacnum1 = dfsigbac['bacnum'][evnum]
plt.title(F'event number: {evnum} \n \
            This event has {signum1} signal hits and {bacnum1} bacground hits', fontsize=16)
plt.legend(loc=5, bbox_to_anchor=(0.2, 0.98), fontsize=14)

fig = plt.figure(figsize=(24, 16)) 
fig.add_subplot(322, projection='polar')
plt.plot(dfsigalotnoise.loc[dfsigalotnoise.event == evnum]['angle'], 5 + dfsigalotnoise.loc[dfsigalotnoise.event == evnum]['layer'], \
          '.', label = 'signal', markersize=8)
plt.plot(dfbacalotnoise.loc[dfbacalotnoise.event == evnum]['angle'], 5 + dfbacalotnoise.loc[dfbacalotnoise.event == evnum]['layer'], \
          '.', label = 'background', markersize=8, color='red')
signum2 = dfsigalotnoise.groupby(['event']).size()[evnum]
bacnum2 = dfbacalotnoise.groupby(['event']).size()[evnum]
plt.title(F'event number: {evnum} when above 1000 hit IDs are also considered as noise\n \
            This event has {signum2} signal hits and {bacnum2} bacground hits', fontsize=16)
plt.legend(loc=5, bbox_to_anchor=(0.2, 0.98), fontsize=14)

fig = plt.figure(figsize=(24, 16)) 
fig.add_subplot(323, projection='polar')
plt.plot(dfnr_nonoise.loc[dfnr_nonoise.event == evnum]['angle'], 5 + dfnr_nonoise.loc[dfnr_nonoise.event == evnum]['layer'], \
          '.', label = 'signal', markersize=8)
signum3 = dfnr_nonoise.groupby(['event']).size()[evnum]
plt.title(F'event number: {evnum} from no noise monte-carlo data\n \
            This event has {signum3} signal hits.', fontsize=16)
plt.legend(loc=5, bbox_to_anchor=(0.2, 0.98), fontsize=14)

fig.add_subplot(325)
plt.plot(dfsig.loc[dfsig.event == evnum]['layer'], dfsig.loc[dfsig.event == evnum]['cell'], '.', label='signal hits')
plt.plot(dfbac.loc[dfbac.event == evnum]['layer'], dfbac.loc[dfbac.event == evnum]['cell'], '.', label='background hits')
plt.xlabel('layer')
plt.ylabel('cell')
plt.title(F'cell-layer plot for event number {evnum}')
plt.legend()

fig.add_subplot(326)
plt.plot(dfsigalotnoise.loc[dfsigalotnoise.event == evnum]['layer'], dfsigalotnoise.loc[dfsigalotnoise.event == evnum]['cell'], '.', label='signal hits')
plt.plot(dfbacalotnoise.loc[dfbacalotnoise.event == evnum]['layer'], dfbacalotnoise.loc[dfbacalotnoise.event == evnum]['cell'], '.', label='background hits')
plt.xlabel('layer')
plt.ylabel('cell')
plt.title(F'cell-layer plot for event number {evnum} when above 1000 hit IDs are also considered noise')
plt.legend()

plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_1/dataanalysis/{t}\
  Polar and cell-layer plots.png', bbox_inches='tight')

################ data preparation for CNN ###############

#########################################################
# normalization and cut in time
j = 20000
maxtime = 1550
traincnnpppipi = np.zeros(shape=(j, 6796), dtype=np.float16)
trvalcnnpppipi = np.zeros(shape=(j, 6796), dtype=np.float16)

for i in range(0, ste[19997]):
    if dfstrarray['time'][i] < maxtime:
        traincnnpppipi[dfstrarray['event'][i], dfstrarray['gid'][i]] = \
        (500 + dfstrarray['time'][i])/(500 + maxtime)
        if dfstrarray['hitid'][i] == 1:
            trvalcnnpppipi[dfstrarray['event'][i], dfstrarray['gid'][i]] = \
            (500 + dfstrarray['time'][i])/(500 + maxtime)

print('\ntraining set for cnn', traincnnpppipi, '\nshape', trvalcnnpppipi.shape, \
   '\ntrue value set for cnn', trvalcnnpppipi, '\nshape', trvalcnnpppipi.shape)

'''print('\npercent of the signal hits with the time values higher than the cutoff (1550)', \
  dfstrarray[dfstrarray.time > 1550]['hitid'].sum() / dfstrarray['hitid'].sum())

print('\npercent of all the hits with time values higher than the cutoff (1550)', \
  (dfstrarray['time'] > 1550).sum() / dfstrarray['hitid'].sum())'''

#########################################################
# prepare events on the rectangle
def sitonsquare(event):
    sqevent = np.zeros(shape=(43, 288))
    wiresum = np.zeros(shape = 44, dtype=int)
    for i in range(43):
        wiresum[i + 1]= np.cumsum(wires)[i]
        w = int(wires[i] / 2)
        for j in range(-w, w):
            sqevent[i, j + 144] = event[(wiresum[i]) + j + w]
    return sqevent

traincnnpppipi_rectangle = np.zeros(shape=[20000, 43, 288])
for i in range(0, 20000):
  traincnnpppipi_rectangle[i] = sitonsquare(traincnnpppipi[i])
print('\ntraincnnpppipi_rectangle', traincnnpppipi_rectangle, '\nshape', traincnnpppipi_rectangle.shape, '\nsum', traincnnpppipi_rectangle.sum())

np.save('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/files/traincnnpppipi_rectangle.npy', traincnnpppipi_rectangle)
np.save('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/files/traincnnpppipi.npy', traincnnpppipi)
np.save('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/files/trvalcnnpppipi.npy', trvalcnnpppipi)

