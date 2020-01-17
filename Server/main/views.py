from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context, loader
from django.views.decorators.csrf import csrf_exempt
from numpy import unique
import urllib.request
from . import function
import cv2

root_url = "http://192.168.0.10"

def sendRequest(url):
	n = urllib.request.urlopen(url)


def homepage(request):
	if request.method == 'POST':
		destination = request.POST.get('destination','')
		img = cv2.imread('static/images/track.jpg')
		im,path = function.djikstra(img,[0,0],destination)

		for p in path:
			sendRequest(root_url+'/'+p)
		'''#urllib.request.urlopen(root_url).read()
		sendRequest(root_url+'/'+destination)'''
	return render(request=request,template_name="index.html"
				  )

'''
def receive(request):
	if request.method == 'POST':
		destination = request.POST.get('destination','')
		#sendRequest(root_url+'/'+destination)
		return HttpResponse(destination)'''
	