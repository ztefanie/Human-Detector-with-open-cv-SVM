# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt


#file = open("firstDETdata.txt", "r")
#for line in file:     
 #   x = float(line)


with open("DETdata_first.txt", "r") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

x = 0
while x < len(content):
    print(content[x+1] + " " + content[x+2])
    plt.plot(content[x+2], content[x+1], 'ro')
    x += 3

f.close

with open("DETdata_retrained.txt", "r") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

x = 0
while x < len(content):
    print(content[x+1] + " " + content[x+2])
    plt.plot(content[x+2], content[x+1], 'bs')
    x += 3


plt.axis([0, 0.1, 0, 0.5])
plt.show()
plt.savefig('myfig.png')

f.close

# evenly sampled time at 200ms intervals
#t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
#plt.plot(t, t, 'r--')
#plt.plot(t, t**2, 'bs')
#plt.plot(t, t**3, 'g^')

#plt.show()

#plt.savefig('myfig.png')