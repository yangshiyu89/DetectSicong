#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:42:02 2018

@author: aaron
"""

import cv2
from matplotlib import pyplot as plt
import timeit

start = timeit.default_timer()
img = cv2.imread('img/ducks.jpg')
template = cv2.imread('img/wangsicong.jpg')
template = cv2.resize(template, (36, 36), interpolation=cv2.INTER_CUBIC)
w, h = template[:,:,0].shape[::-1]

# Apply template Matching
res = cv2.matchTemplate(img, template, eval('cv2.TM_CCOEFF'))
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
imgplt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.rectangle(imgplt,top_left, bottom_right, 255, 2)

plt.imshow(imgplt)
plt.title('Detected results'), plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('img/detected.jpg', cv2.cvtColor(imgplt, cv2.COLOR_BGR2RGB))

stop = timeit.default_timer()
print('Time: ', stop - start)
