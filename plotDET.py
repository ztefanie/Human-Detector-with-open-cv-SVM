# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.backends.backend_tkagg
import sys
import matplotlib.patches as mpatches


fig, ax = plt.subplots()

ytickvalues = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
	
plt.xlabel('FPPW')
plt.title('DET')
plt.ylabel('miss rate')
#plt.axis([0.000001, 0.1, 0.01, 0.5])
plt.axis([0.0001, 0.1, 0.01, 0.5])

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_yticks(ytickvalues)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())



#Draw DET of First-SVM
with open("DETdata_first.txt", "r") as f1:
    content1 = f1.readlines()

x = 0
content1 = [x.strip() for x in content1] 

xlist1 = list(range(0,0))
ylist1 = list(range(0,0))

while x < len(content1):
   #print(content1[x+1] + " " + content1[x+2])
   #plt.plot(content1[x+2], content1[x+1], 'ro')
   xlist1.append(content1[x+2])
   ylist1.append(content1[x+1])
   x += 3

plt.plot(xlist1, ylist1, color='orange')
f1.close


#Draw DET of retrained-SVM
with open("DETdata_retrained.txt", "r") as f2:
    content2 = f2.readlines()

y = 0
content2 = [y.strip() for y in content2] 

xlist2 = list(range(0,0))
ylist2 = list(range(0,0))

while y < len(content2):
    #print(content2[y+1] + " " + content2[y+2])
    #plt.plot(content2[y+2], content2[y+1], 'bs')
    xlist2.append(content2[y+2])
    ylist2.append(content2[y+1])
    y += 3


plt.plot(xlist2, ylist2, color='blue')

firstlabel = mpatches.Patch(color='blue', label='first SVM')
retrainedlabel = mpatches.Patch(color='orange', label='retrained SVM')
plt.legend(handles=[firstlabel, retrainedlabel])

f2.close


plt.savefig('DET.png')