from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context, loader
from django.views.decorators.csrf import csrf_exempt
from numpy import unique
import urllib.request
root_url = "http://192.168.0.102"

def sendRequest(url):
	n = urllib.request.urlopen(url)

def get_data():
	global data

	n = urllib.request.urlopen(url).read() # get the raw html data in bytes (sends request and warn our esp8266)
	n = n.decode("utf-8") # convert raw html bytes format to string :3
	
	data = n
	
def homepage(request):
	if request.method == 'POST':
		destination = request.POST.get('destination','')
		sendRequest(root_url+'/'+destination)
		get_data()
	return render(request=request,template_name="index.html"
				  )