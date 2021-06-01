import numpy as np
import pylab as pl
from matplotlib import collections as mc
import random

class Line:
	def __init__(self, x1, y1, x2, y2):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

		if x1 == x2:
			self.isVert = 1
		else:
			self.isVert = 0

	def crosses(self, l):
		## If they are both horizontal or both vertical then no
		## It is assumed that they are not colinear
		if self.isVert == l.isVert:
			return 0

		if self.isVert:
			## If the low y value is above that of horz then they don't
			if self.y1 > l.y2:
				return 0

			## If the high y value is below that of horz then then they don't
			if self.y2 < l.y1:
				return 0

			## Now check the x values
			if self.x2 < l.x1:
				return 0

			if self.x2 > l.x2:
				return 0
		else:
			if l.y1 > self.y2:
				return 0

			if l.y2 < self.y1:
				return 0

			## Now check the x values
			if l.x2 < self.x1:
				return 0

			if l.x2 > self.x2:
				return 0

		return 1

	def print(self):
		print("(%d, %d), (%d, %d)" %(self.x1, self.y1, self.x2, self.y2))

	def isColinear(self, l):
		if self.isVert != l.isVert:
			return 0

		if self.isVert:
			## If the x values are not the same then they cannot be colinear
			if ( self.x1 != l.x1):
				return 0

			## If the start of one is between the start and finish of the other then yes
			if self.y1 <= l.y1 <= self.y2:
				return 1

			## If the finish of one is between the start and finish of the other then yes
			if self.y1 <= l.y2 <= self.y2:
				return 1

			## If the start of one is between the start and finish of the other then yes
			if l.y1 <= self.y1 <= l.y2:
				return 1

			## If the finish of one is between the start and finish of the other then yes
			if l.y1 <= self.y2 <= l.y2:
				return 1

		else:
			## If the y values are not the same then they cannot be colinear
			if self.y1 != l.y1:
				return 0

			## If the start of one is between the start and finish of the other then yes
			if self.x1 <= l.x1 <= self.x2:
				return 1

			## If the finish of one is between the start and finish of the other then yes
			if self.x1 <= l.x2 <= self.x2:
				return 1

			## If the start of one is between the start and finish of the other then yes
			if l.x1 <= self.x1 <= l.x2:
				return 1

			## If the finish of one is between the start and finish of the other then yes
			if l.x1 <= self.x2 <= l.x2:
				return 1

		return 0

def AddPoint(plot, x, y, color):
    plot.scatter(x, y, c=color, s=1)

## If the x values are the same with each point then it is vertical
def isVertical( line ):

	assert len(line) == 2

	if ( line[0][0] == line[1][0]):
		return 1

	return 0

def isColinear(Lines, line):
	for l in Lines:
		if line.isColinear(l):
			return 1

	return 0

## Generated lines will not be colinear
def genLines():
	topright = 100

	## generate some lines

	horz = []
	vert = []

	## some horizontal
	while len(horz) < 20:
	    x1 = random.randint(0, topright * .6)
	    x2 = random.randint(x1, topright)
	    y = random.randint(0, topright)

	    ## Make sure that the x1 and x2 are not the same
	    ## This is vor visibility on the plot but we also don't want single point lines
	    if ( x1 == x2):
	    	continue

	    if abs(x1 - x2) < 3:
	    	continue

	    l = Line(x1, y, x2, y)

	    if isColinear(horz, l):
	    	continue

	    horz.append(l)

	#some vertical
	while len(vert) < 20:
	    y1 = random.randint(0, topright * .6)
	    y2 = random.randint(x1, topright)
	    x = random.randint(0, topright)

	    if (y1 == y2):
	    	continue

	    if abs(y1 - y2) < 3:
	    	continue

	    l = Line(x, y1, x, y2)

	    if isColinear(vert, l):
	    	continue

	    vert.append(l)

	return horz + vert

def linesToVertices( Lines ):
	V = []
	E = []

	## This array is used for vertices of specific lines.
	## I use it later to create the edges
	Vl = []

	for i in range(len(Lines)):
		Vl.append([])

	epsilon = .25

	for i in range(len(Lines)):
		for j in range(i+1, len(Lines)):
			## Do they intersect

			if Lines[i].crosses(Lines[j]) == 0:
				continue

			if Lines[i].isVert:
				x = Lines[i].x1
				y = Lines[j].y1
			else:
				x = Lines[j].x1
				y = Lines[i].y1

			## Append the 4 corners
			V.append([x - epsilon, y + epsilon])
			V.append([x + epsilon, y + epsilon])
			V.append([x - epsilon, y - epsilon])
			V.append([x + epsilon, y - epsilon])

			## Each line needs the vertices too
			Vl[i].append([x - epsilon, y + epsilon])
			Vl[i].append([x + epsilon, y + epsilon])
			Vl[i].append([x - epsilon, y - epsilon])
			Vl[i].append([x + epsilon, y - epsilon])

			Vl[j].append([x - epsilon, y + epsilon])
			Vl[j].append([x + epsilon, y + epsilon])
			Vl[j].append([x - epsilon, y - epsilon])
			Vl[j].append([x + epsilon, y - epsilon])

	## Add the vertices to the end of line points
	for i in range(len(Lines)):
		if Lines[i].isVert:
			# Bottom point. I used 2*epsilon to see better and help with edge creation
			V.append( [Lines[i].x1, Lines[i].y1 - 2*epsilon])
			Vl[i].append( [Lines[i].x1, Lines[i].y1 - 2*epsilon])

			# Top point
			V.append( [Lines[i].x1, Lines[i].y2 + 2*epsilon])
			Vl[i].append([Lines[i].x1, Lines[i].y2 + 2*epsilon])

		else:
			## Left point
			V.append( [Lines[i].x1 - 2*epsilon, Lines[i].y1])
			Vl[i].append( [Lines[i].x1 - 2*epsilon, Lines[i].y2])

			## Right point
			V.append( [Lines[i].x2 + 2*epsilon, Lines[i].y2])
			Vl[i].append( [Lines[i].x2 + 2*epsilon, Lines[i].y2])

	## Time to create the Edges
	for l in range(len(Lines)):
		verts = Vl[l]

		if ( Lines[l].isVert ):
			## Sort by y coordinate
			verts.sort(key=lambda x:x[1])
			if len(verts) == 2:
				E.append( [ verts[0], verts[1] ] )
				E.append( [ verts[1], verts[0] ] )
			else:
				## Left two edges
				E.append([ verts[0], verts[1] ])
				E.append([ verts[0], verts[2] ])

				## Right two edges
				E.append([ verts[-1], verts[-2] ])
				E.append([ verts[-1], verts[-3] ])

				start = 3
				end = len(verts) - 3

				for i in range(start, end, 4):
					E.append( [verts[i], verts[i+2]] )
					E.append( [verts[i+1], verts[i+3]] )

		else:
			## Sort by x coordinate
			verts.sort(key=lambda x:x[0])

			if len(verts) == 2:
				E.append( [ verts[0], verts[1] ] )
				E.append( [ verts[1], verts[0] ] )
			else:
				## Left two edges
				E.append([ verts[0], verts[1] ])
				E.append([ verts[0], verts[2] ])

				## Right two edges
				E.append([ verts[-1], verts[-2] ])
				E.append([ verts[-1], verts[-3] ])

				start = 3
				end = len(verts) - 3

				for i in range(start, end, 4):
					E.append( [verts[i], verts[i+2]] )
					E.append( [verts[i+1], verts[i+3]] )
	
	return V, E


random.seed(12345)
lines = genLines()

'''
lines = []
lines.append(Line(20, 20, 40,20))
lines.append(Line(25, 15, 25,25))
lines.append(Line(35, 15, 35,25))
'''

l = []

for x in lines:
	l.append([ [x.x1, x.y1], [x.x2, x.y2]])

#l = [ [[20, 20], [40,20]], [[25,15], [25, 25]], [[35,15], [35, 25]]   ]
lc = mc.LineCollection(l, linewidths=1)
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.margins(0.1)

V,E = linesToVertices(lines)

#e = []
#for x in E:
#	e.append([ [x.x1, x.y1], [x.x2, x.y2]])

le = mc.LineCollection(E, linewidths=1)
ax.add_collection(le)

for x in V:
	AddPoint(ax, x[0], x[1], 'black')

pl.show()
