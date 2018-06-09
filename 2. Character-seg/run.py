import glob
import os

listOfFiles=glob.glob('*.jpg')

for i in listOfFiles:
	os.system('python3 Charseg.py '+i)	