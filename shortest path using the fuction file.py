import numpy as np
import cv2
import function
import urllib.request
import time
root_url = "http://192.168.0.102"

 
def sendRequest(url):
	n = urllib.request.urlopen(url)
 


img = cv2.imread('./Tracks/track-iot-bot-2x2-red-blue.jpg')
im,path = function.djikstra(img,[0,0],"10")
print(path)
#for p in path:
#	sendRequest(root_url+'/'+p)
	
cv2.imshow('image',im)

cv2.waitKey(0)
cv2.destroyAllWindows()