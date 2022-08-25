# openCV contour rotation and scale about either center of mass of arbitrary point
# https://math.stackexchange.com/questions/3245481/rotate-and-scale-a-point-around-different-origins
# i wanted to scale up a contour w/o having to use images and morphological dilate. 

import numpy as np
import cv2
image = np.zeros((225,225), np.uint8)  

cv2.rectangle(image, (55, 55), (120, 120), 255, -1)

contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# results of these operations are floats, cast them to ints however you see fit.
def scaleRotateContourAboutPoint(contour, scaleTuple, angle, customPoint = []):

    angle = angle*np.pi/180

    if len(customPoint) == 2: (cx,cy) = customPoint
    else: (cx,cy) = (lambda m: (m['m10']/m['m00'], m['m01']/m['m00'])) (cv2.moments(contours[0])) #
    
    (sx,sy)  = scaleTuple

    MTcenter = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
    MRotate = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    MScale = np.array([[sx,0,0],[0,sy,0],[0,0,1]])
    MTback = np.array([[1,0,cx],[0,1,cy],[0,0,1]])

    M  = np.linalg.multi_dot([MTback,MScale,MRotate,MTcenter])

    cntr = np.array([np.matmul(M,[x,y,1])[:2] for [[x,y]] in contour],np.int32).reshape(-1,1,2)
    return cntr, np.uint32((cx,cy))


(sx,sy) = (2,1) 
angle = 0 #degrees
cntr, (cx,cy) = scaleRotateContourAboutPoint(contours[0], (sx,sy) ,0, customPoint = [])


cv2.drawContours( image, [cntr], -1, 175, 2) 
cv2.circle(image,(cx,cy), 2, 120, -1)
cv2.imshow('anis-scale + rot about center', image) 

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()

