from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tkinter import *
from PIL import Image,ImageGrab

mnist=fetch_openml('MNIST_784')

x=mnist['data']
y=mnist['target']
x=np.array(x)
some_digit=x[36007]
some_digit_img=some_digit.reshape(28,28)

xtrain,xtest,ytrain,ytest=train_test_split(x,y)

ytrain=ytrain.astype(int)
ytest=ytest.astype(int)

ytrain_2=(ytrain==9)
ytest_2=(ytest==9)

log=LogisticRegression()

log.fit(xtrain,ytrain_2)

print(log.predict([some_digit]))

root=Tk()
root.title("character recognition")

def getx_y(event):
	global x1,y1
	x1,y1=event.x,event.y
def move_c(event):
	global x1,y1
	can.create_line((x1,y1,event.x,event.y),fill="white",width=5)
	x1,y1=event.x,event.y
def dele():
	can.delete("all")
def recognise():
	global xtrain,ytrain,ytrain_2,log,some_digit
	img=ImageGrab.grab(bbox=(172,150,984,962))#left bottom right top
	img.save("img.png","png")
	img=img.resize((28,28))
	img=img.convert("L")
	img=img.convert("1")
	img=np.array(img)
	img=img.reshape(1,-1,28,28)
	img=img.flatten()
	print(log.predict([img]))

can=Canvas(root,width=400,height=400,bg="black")
can.pack()
can.bind("<Button-1>",getx_y)
can.bind("<B1-Motion>",move_c)

b=Button(root,text="delete",command=dele)
b.pack()
sub=Button(root,text="enter",command=recognise)
sub.pack()

root.mainloop()
