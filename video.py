import numpy as np
import cv2
import function
import time
 


camera = cv2.VideoCapture("http://192.168.0.102:4747/mjpegfeed?900x900'")

 
# otherwise, grab a reference to the video file
while True:
    (grabbed, frame) = camera.read()
    #img = cv2.resize(frame, (804,797))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("track.jpg")
    im,result = function.djikstra(template,"11")
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()