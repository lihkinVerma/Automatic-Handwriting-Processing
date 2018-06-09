# using speech library
import os
import speech as s

os.system("tesseract hlo.jpg aut")

f=open('aut.txt')
a=f.read()
s.say(a)
