import numpy as np
import cv2
import function
import time
import imutils
import math


 

#img = cv2.resize(frame, (804,797))
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
template = cv2.imread("track.jpg")

 

#dictionary of all contours
contours = {}
#array of edges of polygon
approx = []
#scale of the text
scale = 2
#camera
camera = cv2.VideoCapture("http://192.168.0.3:4747/mjpegfeed?'804x797")
#img = cv2.resize(img, (804,797))

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#calculate angle
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)


    

    
while True:
    (grabbed, frame) = camera.read()
    
    # define the lower and upper boundaries of the colors in the HSV color space
    lower = { 'green':(40, 100, 50), 'blue':(97, 100, 117)} #assign new item lower['blue'] = (93, 10, 0) 'yellow':(54,255,255)
    upper = {'green':(80,255,255), 'blue':(117,255,255)}
    colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0)} 
    # define standard colors for circle around the object
    #colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}
     
       
     
    
    
    
    
    frame = imutils.resize(frame, height=904,width=897)
     
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #for each color in dictionary check object in frame
    
    #grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Canny
    canny = cv2.Canny(gray,80,240,3)
    
    #contours
    canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):
        #approximate the contour with accuracy proportional to
        #the contour perimeter
        approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)
    
        #Skip small or non-convex objects
        if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
            continue
    
        #triangle
        if(len(approx) == 3):
            x,y,w,h = cv2.boundingRect(contours[i])
            #cv2.putText(frame,'TRI',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
        elif(len(approx)>=4 and len(approx)<=6):
            #nb vertices of a polygonal curve
            vtc = len(approx)
            #get cos of all corners
            cos = []
            for j in range(2,vtc+1):
                cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
            #sort ascending cos
            cos.sort()
            #get lowest and highest
            #mincos = cos[0]
            #maxcos = cos[-1]
    
            #Use the degrees obtained above and the number of vertices
            #to determine the shape of the contour
            x,y,w,h = cv2.boundingRect(contours[i])
            #if(vtc==4):
                #cv2.putText(frame,'RECT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
            #elif(vtc==5):
                #cv2.putText(frame,'PENTA',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
            #elif(vtc==6):
                #cv2.putText(frame,'HEXA',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
        else:
            #detect and label circle
            #area = cv2.contourArea(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            radius = w/2
            #if(abs(1 - (float(w)/h))<=2 and abs(1-(area/(math.pi*radius*radius)))<=0.2):
                #cv2.putText(frame,'CIRC',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
        
    
    for key, value in upper.items():
        # construct a mask for the color from dictionary`1, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
               
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        #center = None
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #M = cv2.moments(c)
            #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
       
            # only proceed if the radius meets a minimum size. Correct this value for your obect's size
            if radius > 0.5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                #cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                print(x,y)
                #cv2.putText(frame,key , (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                if key=='blue':
                    
                    #im,result = function.djikstra(template,[x,y],"11")
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                    cv2.putText(frame,key , (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                    print(x+30,y+60)
    cv2.imshow("Frame", frame)
   
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
                #Capture frame-by-frame
    #Display the resulting frame 
# show the frame to our screen

