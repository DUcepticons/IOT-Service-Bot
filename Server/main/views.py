from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context, loader
from django.views.decorators.csrf import csrf_exempt
from numpy import unique

def homepage(request):
	
	return render(request=request,template_name="index.html"
				  )