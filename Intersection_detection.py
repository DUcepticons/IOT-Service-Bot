# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:34:52 2020

@author: Riad
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
img = cv2.imread('track_3.jpg')
#im = cv2.resize(img, (960, 540))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresh_type = cv2.THRESH_BINARY_INV
bin_img = cv2.adaptiveThreshold(edges, 255, adapt_type,thresh_type, 11, 2)

rho, theta, thresh = 1, np.pi/180, 500
lines = cv2.HoughLines(bin_img, rho, theta, thresh)

from collections import defaultdict
def segment_by_angle_kmeans(lines, k=3, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 0.1))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

segmented = segment_by_angle_kmeans(lines)

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections



point=[]
y_list=[]
intersections = segmented_intersections(segmented)
count=0
for intr in intersections:
    point.append(intr[0])
    #cv2.circle(img, (int(intr[0][0]), int(intr[0][1])), 1, (255, 0, 0),3)
font = cv2.FONT_HERSHEY_SIMPLEX 
poi=np.asarray(point)
#plt.scatter(poi[:, 0], poi[:, 1])
kmeans = KMeans(n_clusters=17)
kmeans.fit(point)
cent=kmeans.cluster_centers_
#plt.scatter(cent[:, 0], cent[:, 1], c='black', s=200, alpha=0.5)

poi=[]
for p in cent:
    # collect actual node location
    if 900>=p[0]>=130 and 900>=p[1]>=130: #by testing the graph position is between 130x700
        poi.append(p)
i=0
for c in poi:
    cv2.circle(img, (int(c[0]), int(c[1])), 30, (0, 0, 255),3)
    img = cv2.putText(img, str(i), (int(c[0]), int(c[1])), font,  
                       1, (255, 0, 0), 2, cv2.LINE_AA)
    i+=1
print("Total Node: ",i)
im = cv2.resize(img, (540, 540))
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()