#!/usr/bin/env python
# coding: utf-8

# In[1]:

#test
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# In[2]:


def median_filter(src, ksize):
    d = int((ksize-1)/2)
    h, w = src.shape[0], src.shape[1]
    

    src1 = np.pad(src, [d,d], mode='symmetric')
    
    dst = src.copy()

    for y in range(0, h):
        for x in range(0, w):

            dst[y][x] = np.median(src1[y:y+2*d+1, x:x+2*d+1])

    return dst


# In[3]:


import itertools
def median_kai6(src, ksize,t,N):
    d = int((ksize-1)/2)
    h, w = src.shape[0], src.shape[1]
    
    dst = src.copy()
    for i in range(N):
        src1 = np.pad(dst, [d,d], mode='symmetric')
        for y in range(0, h):
            for x in range(0, w):
                for k in dst[y][x]:
                    if k==0 or k==255:    
                        p=src1[y:y+2*d+1, x:x+2*d+1]
                        q=p.ravel()
                        n1 = []                 
                        for i in q:
                            if i != 0 and i != 255: 
                                n1.append(i)
                        if abs(k-np.median(n1))>t:
                            dst[y][x] = np.mean(n1)
    return dst


# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('idata4.bmp')

img2 = cv2.imread('rdata4.bmp')

x=median_kai6(img,5,5,2)

plt.imshow(x),plt.title('median_kai6')
plt.xticks([]), plt.yticks([])

plt.show()

a=mse(img2,x)

print('median_kai6_MSE:')
print(a)


# In[ ]:




