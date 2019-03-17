#  ARROW DIRECTION IDENTIFIER
#
#  Author: Nilay Savant
#
#  Description: Identifies direction of arrow image (in construction!)
#
#

import cv2
import numpy as np
import time
from math import atan
from math import degrees
from math import atan2
from math import sqrt

# CUSTOM GLOBAL VARIABLES

FRAME_COUNT = 2  # no of frames to analyse for mean per cycle

direction = []  # to store direction angle for each frame

# Initialise the Arrow cascade classifier
arrow_cascade = cv2.CascadeClassifier('cascades/arrow_cascade_stage_11.xml')

# CUSTOM FUNCTIONS : ---------------------------------------------------<<<<<<<<<<<<<<<<<<<<<<


def displayText(frame, text, fontColor = (0, 0, 255), bottomLeftCornerOfText = (10, 100)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (10, 500)
    fontScale = 2
    
    lineType = 3

    cv2.putText(frame, text, bottomLeftCornerOfText, font, fontScale,
                fontColor, lineType)


# function for average
def average(list_in):
    return (sum(list_in) / len(list_in))


# Get perpendicular dist between a point (x,y) from a line defined by (x1,y1) and (x2,y2)
def distFromLine(x1, y1, x2, y2, x, y):
    distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 -
                   y2 * x1) / sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2))
    return distance


# To sort list 'lst' in ascending order and return sorted and original hyarchy index list
def sortAscending(lst):
    index = range(len(lst))
    mx = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[j] < lst[i]:
                tempVal = lst[i]
                lst[i] = lst[j]
                lst[j] = tempVal

                tempInd = index[i]
                index[i] = index[j]
                index[j] = tempInd

    return lst, index


def getArrowDirection(frame):
    # CUSTOM VARIABLES AND LISTS

    distances = []  # to store raw distances from line

    approx = []  # to store vertices of arrow polygon

    sorted_distances = []  # to store sorted distances

    org_dist_ind = []  # to store original index hyarchy of initial distances

    angle = 360

    ## frame = cv2.imread('arrow_right_pdf.png', 0)

    # invert img (since we want black object) ##
    # frame_inv = cv2.bitwise_not(frame)

    # blur img
    blur = cv2.blur(frame, (3, 3))  #(2,2)  # (1,1) on the raspi

    #cv2.imshow('blur', blur)
    # threshold image and invert it
    ret, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

    #cv2.imshow('thresh', thresh)

    cv2.imshow('result', thresh)

    # thresh = cv2.bilateralFilter(thresh,1,75,75) #10,75,75

    # find contours in thresh img
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (100,100,100), 3)
    #cv2.imshow('CHE', im2)

    HEIGHT, WIDTH = im2.shape

    # Find contour matching 7 vertices
    for c in contours:
        distances = []
        area = cv2.contourArea(c)
        if area > 200:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cy < ((HEIGHT / 2) + 50) and cy > (
                    (HEIGHT / 2) - 50) and cx < ((WIDTH / 2) + 70) and cx > (
                        (WIDTH / 2) - 70):
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.01 * peri, True)  # 0.04
                    if (len(approx) == 7):
                        # print ")))))))))))))))))))))))))))))))))))))))))))))))))))))))))0000"

                        vert_num = len(approx)
                        # print approx
                        # print approx
                        # cv2.drawContours(frame, [c], -1, (6,255,0), 6)

                        cv2.drawContours(frame, contours, -1, (100, 100, 100),
                                         3)

                        # then apply fitline() function
                        [vx, vy, x, y] = cv2.fitLine(approx, cv2.DIST_L2, 1,
                                                     0.01, 0.01)

                        # Now find two extreme points on the line to draw line
                        lefty = int((-x * vy / vx) + y)
                        righty = int(((frame.shape[1] - x) * vy / vx) + y)

                        # Finally draw the line
                        if frame.shape[1] > 0 and righty > 0 and lefty > 0:
                            cv2.line(frame, (frame.shape[1] - 1, righty),
                                     (0, lefty), 255, 2)
                        else:
                            break

                        # Get dist for each vertice from mean line (passing through the arrow) and save it to distances list
                        for i in range(vert_num):
                            distances.append(
                                distFromLine(frame.shape[1] - 1, righty, 0,
                                             lefty, approx[i][0][0],
                                             approx[i][0][1]))

                        # get sorted dist and hyerchy index
                        sorted_distances, org_dist_ind = sortAscending(
                            distances)

                        # get x and y cord of tip of arrow
                        x_tip = approx[org_dist_ind[0]][0][0]
                        y_tip = approx[org_dist_ind[0]][0][1]

                        cv2.circle(frame, (x_tip, y_tip), 10, (100, 100, 100),
                                   -1)

                        # get x and y of approx adjacent to the arrow_tip
                        x_end1 = approx[org_dist_ind[len(org_dist_ind) -
                                                     1]][0][0]
                        y_end1 = approx[org_dist_ind[len(org_dist_ind) -
                                                     1]][0][1]
                        x_end2 = approx[org_dist_ind[len(org_dist_ind) -
                                                     2]][0][0]
                        y_end2 = approx[org_dist_ind[len(org_dist_ind) -
                                                     2]][0][1]

                        # calculate mean of adjacent vetices
                        x_mean = (x_end1 + x_end2) / 2
                        y_mean = (y_end1 + y_end2) / 2

                        # display the mean vertice
                        # print x_mean, y_mean
                        # draw mean vertice on frame
                        cv2.circle(frame, (x_mean, y_mean), 10,
                                   (100, 100, 100), -1)

                        angle = degrees(atan2(y_tip - y_mean, x_tip - x_mean))

                        # if angle >= -90 and angle < 90:
                        # print " >>>>>>>>>>>>>>>>>>>>>>>>>>>> RIGHT ARROW >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------------------------------"

                        # elif angle <= -90 or angle > 90:
                        # print " <<<<<<<<<<<<<<<<<<<<<<<<<<<< LEFT ARROW --------------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                        cv2.imshow('CHECK', frame)
                        # break
    return frame, angle


cap = cv2.VideoCapture(0)  # video capture device object

X = 0
Y = 0
W = 0
H = 0

arrows = []

while (True):

    frame_num = 0  # to iterate over frames
    #
    right = 0
    left = 0
    none = 0

    # Capture FRAME_COUNT no of frames analyse each and take avg at 30FPS
    while (frame_num < FRAME_COUNT):

        direction_angle = 360

        # Capture frame-by-frame form camera
        ret, frame = cap.read()

        # convert frame to garyscale
        frame_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect arrows in the frame using arrow_cascde(haar classifier),
        # 'arrows' stores a list of rectangles(bounding the arrow) returned by the function
        arrows = arrow_cascade.detectMultiScale(frame_conv, 1.2,
                                                44)  #(1.08, 44)

        if len(arrows) != 0:
            for (x, y, w, h) in arrows:
                arrow_detected = frame_conv[y:(y + h), x:(
                    x + w)]  # Crop from y1:y2, x1:x2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                fra, direction_angle = getArrowDirection(arrow_detected)
                #cv2.imshow('aa', arrow_detected)
                #cv2.imshow('fra', fra)

        if direction_angle == 360:
            none += 1  # incrment none
            #print "NONE____________________________________________________________________________________0000_____________"
        elif direction_angle >= -90 and direction_angle < 90:
            right += 1  # incrment right
            #print " >>>>>>>>>>>>>>>>>>>>>>>>>>>> RIGHT ARROW >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------------------------------"

        elif direction_angle <= -90 or direction_angle > 90:
            left += 1  # incrment left
            #print " <<<<<<<<<<<<<<<<<<<<<<<<<<<< LEFT ARROW --------------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

        # increment frame_num
        frame_num = frame_num + 1

        # show frame
        cv2.imshow('Frame', frame)

    # if none is detected more than left or right display none
    if none > left and none > right:
        print "NONE____________________________________________________________________________________0000_____________"

    # if left is greater
    elif left > none and left > right:
        print " <<<<<<<<<<<<<<<<<<<<<<<<<<<< LEFT ARROW --------------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        displayText(frame, 'LEFT', (255,0,0))      

    # if right is greater
    elif right > none and right > left:
        print " >>>>>>>>>>>>>>>>>>>>>>>>>>>> RIGHT ARROW >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------------------------------"
        displayText(frame, 'RIGHT', (0,0,255))

    cv2.imshow('Frame', frame)
    # To exit from loop press 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
