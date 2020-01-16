import numpy as np
import cv2
import function
 
 

 


camera = cv2.VideoCapture("http://192.168.0.101:4747/mjpegfeed?640x480'")
   
 
# otherwise, grab a reference to the video file
while True:
    (grabbed, frame) = camera.read()
    im,result = function.djikstra(frame,"11")
    cv2.imshow("Frame", im)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()