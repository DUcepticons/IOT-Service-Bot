from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context, loader
from django.views.decorators.csrf import csrf_exempt
from numpy import unique
import urllib.request
from . import function

root_url = "http://192.168.0.13"

def sendRequest(url):
	n = urllib.request.urlopen(url)


def homepage(request):
	if request.method == 'POST':
		destination = request.POST.get('destination','')
		'''img = cv2.imread('track-iot-bot-2x2-red-blue.jpg')
		im,path = function.djikstra(img,[0,0],"10")

		for p in path:
			sendRequest(root_url+'/'+p)'''
		#urllib.request.urlopen(root_url).read()
		sendRequest(root_url+'/'+destination)
	return render(request=request,template_name="index.html"
				  )

'''
def receive(request):
	if request.method == 'POST':
		destination = request.POST.get('destination','')
		#sendRequest(root_url+'/'+destination)
		return HttpResponse(destination)'''
	