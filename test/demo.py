import sys
sys.path.append('../src')
import Model
import math
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure

test = Model.word2vec('skipgram', 3, 3, 'fruit.txt', True, 0, True)

total_before = datetime.now()

# before
fig = figure('word2vec demo')
ax = Axes3D(fig)

for i in range(test.voc.size()): #plot each point + it's index as text above
	ax.scatter(test.W_i[i, 0], test.W_i[i, 1], test.W_i[i,2], color='b') 
	ax.text(test.W_i[i, 0], test.W_i[i, 1], test.W_i[i,2], '%s' % (test.voc.at(i)), size=10, zorder=1, color='k') 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

print('similar to apple', test.similarTo('apple'))

for i in range(2000):
	#print('===before===')
	before = datetime.now()
	test.train(0.025)
	after = datetime.now()
	delta = after - before
	#print('===after====')
	if(i % 100 == 0):
		print("[", i, "] " + str(delta.seconds) + "s")
		#print(test.W_o)

total_after = datetime.now()

total_delta = total_after - total_before
print("total elapsed time: " + str(total_delta.seconds) + "s")

print('words similar to apple: ', test.similarTo('apple'))

fig = figure('word2vec demo')
ax = Axes3D(fig)

for i in range(test.voc.size()): #plot each point + it's index as text above
	ax.scatter(test.W_i[i, 0], test.W_i[i, 1], test.W_i[i,2], color='b') 
	ax.text(test.W_i[i, 0], test.W_i[i, 1], test.W_i[i,2], '%s' % (test.voc.at(i)), size=10, zorder=1, color='k') 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
