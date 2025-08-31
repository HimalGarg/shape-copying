import tkinter as tk
from PIL import Image, ImageTk
import turtle as t
from task1.shapedetector import *
from task1.centre_of_shape import *
from task1.color_detector import *
import argparse
import imutils
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="path to the input image")

args=vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)[1]

shape_data={}


tur = t.Turtle()
tur.shape("classic")
t.title("Turtle")
t.setup(image.shape[1],image.shape[0])
t.screensize(image.shape[1],image.shape[0])
t.setworldcoordinates(0, image.shape[0], image.shape[1], 0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

def shape_clicked(shape_id):
    print(f"{shape_id} clicked!")
    points = shape_data[shape_id]
    tur.penup()
    tur.goto(points[0])
    tur.pendown()
    for (x,y) in points[1:]:
        tur.goto(x,y)
    tur.goto(points[0])

# Tkinter root window
root = tk.Toplevel()
root.title("Shape Button GUI")



# Load and display image
img = Image.open(args["image"])
photo = ImageTk.PhotoImage(img)

canvas = tk.Canvas(root, width=image.shape[1], height=image.shape[0])
canvas.pack()
canvas.create_image(0, 0, image=photo, anchor="nw")


# Example: define clickable areas (x1, y1, x2, y2)
# You will need to adjust these coordinates to match the shapes in your image
for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.04*peri, True)
    cX, cY = findCentroid(c)
    
    
    (x, y, w, h) = cv2.boundingRect(approx)
    rect = cv2.minAreaRect(approx)
    box = cv2.boxPoints(rect)
    
    poly_id = canvas.create_polygon(box[0][0], box[0][1], box[1][0], box[1][1],box[2][0],box[2][1],box[3][0],box[3][1], outline="", fill="")
    shape_data[poly_id] = approx.reshape(-1, 2)
    canvas.tag_bind(poly_id, "<Button-1>", lambda e, pid=poly_id: shape_clicked(pid))




root.mainloop()
