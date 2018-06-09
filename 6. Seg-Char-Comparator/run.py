import glob as g
import os

lof=g.glob('*.jpg')
n=len(lof)

for i in range(0,n-1):
#	for j in lof:
#		if j is not lof[i]:
	os.system('python PILcompare.py '+lof[i]+' '+lof[i+1])