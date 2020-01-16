from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context, loader
from django.views.decorators.csrf import csrf_exempt
from numpy import unique
import urllib.request
root_url = "http://192.168.0.102"

def sendRequest(url):
	n = urllib.request.urlopen(url)


def homepage(request):
	if request.method == 'POST':
		destination = request.POST.get('destination','')
		sendRequest(root_url+'/'+destination)
		#urllib.request.urlopen(root_url).read()
	return render(request=request,template_name="index.html"
				  )

'''
def receive(request):
	if request.method == 'POST':
		destination = request.POST.get('destination','')
		#sendRequest(root_url+'/'+destination)
		return HttpResponse(destination)'''
	