import cv2 as cv
import cv2.aruco as aruco
import math
import matplotlib.pyplot as mlt
import numpy as np

#finding ids of aruco markers


def findaruco(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    key = getattr(aruco, 'DICT_5X5_250')
    arucodict = aruco.Dictionary_get(key)
    arucoparameter = aruco.DetectorParameters_create()
    (corners, ids,
     rejectedids) = aruco.detectMarkers(gray,
                                        arucodict,
                                        parameters=arucoparameter)

    x1, y1 = corners[0][0][0][0], corners[0][0][0][1]
    x2, y2 = corners[0][0][1][0], corners[0][0][1][1]
    Y = y2 - y1
    X = x2 - x1
    var = Y / X
    slope = math.atan(var) * (180 / 3.14
                              )  #calculating inclination of aruco marker

    lenthside = math.sqrt((Y**2) + (X**2))
    return corners, ids[0][0], ((int(x1)), (int(y1))), slope, lenthside


#function to rotate image


def image_rotation(img, angle, rotationpoint):
    (height, width) = img.shape[:2]

    matrix_ = cv.getRotationMatrix2D(rotationpoint, angle, 1.0)

    #matrix transforms the image by a given angle from the given point

    dimensions = (width, height)
    return cv.warpAffine(img, matrix_, dimensions)


def augmenting(corners, ids, img, imgmain):
    tl = corners[0][0], corners[0][1]
    tr = corners[1][0], corners[1][1]
    br = corners[2][0], corners[2][1]
    bl = corners[3][0], corners[3][1]

    #coordinates of the 4 points of the aruco markers can be found

    h, w, _ = imgmain.shape
    points1 = np.array([tl, tr, br, bl])
    points2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv.findHomography(points2, points1)

    #matrix to transform points 1 to points 2

    img_ = cv.warpPerspective(imgmain, matrix, (img.shape[1], img.shape[0]))

    #warps markers to square boxes

    cv.fillConvexPoly(img, points1.astype(int), (0, 0, 0))
    img_ = img + img_

    return img_


def contourfind(img):
    _, thrash = cv.threshold(img, 220, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True),
                                 True)
        if len(approx) == 4:
            x1, y1, w, h = cv.boundingRect(approx)
            asp_rat = float(w) / h
            if asp_rat >= 0.98 and asp_rat <= 1.02:

                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                return box

                #gives corners of bounding box


IMG = cv.imread("Downloads/CVtask.jpg")
IMG1 = cv.resize(IMG, (0, 0), fx=0.5, fy=0.5)

#reducing image size to make it easier

hsv_img1 = cv.cvtColor(IMG1, cv.COLOR_BGR2HSV)

#black colour
lower = np.array([0, 0, 0])
upper = np.array([10, 210, 210])
black = cv.inRange(hsv_img1, lower, upper)

#peach colour
lower = np.array([4, 4, 216])
upper = np.array([29, 35, 251])
peach = cv.inRange(hsv_img1, lower, upper)

#orange colour
lower = np.array([2, 221, 150])
upper = np.array([18, 255, 255])
orange = cv.inRange(hsv_img1, lower, upper)

#green colour
lower = np.array([20, 51, 40])
upper = np.array([48, 255, 255])
green = cv.inRange(hsv_img1, lower, upper)

# reading the images of aruco markers
img1 = cv.imread("Downloads/3.jpg")
img2 = cv.imread("Downloads/4.jpg")
img3 = cv.imread("Downloads/1.jpg")
img4 = cv.imread("Downloads/2.jpg")
dict1 = []
dict1.append(img1)
dict1.append(img2)
dict1.append(img3)
dict1.append(img4)

#using the functions

for ele in dict1:
    corners, ids, point, s, L = findaruco(ele)
    rotated = image_rotation(ele, s, point)
    transformed = rotated[point[1]:point[1] + int(L),
                          point[0]:point[0] + int(L)]
    if ids == 1:
        box = contourfind(green)
        IMG1 = augmenting(box, ids, IMG1, transformed)
    if ids == 2:
        box = contourfind(orange)
        IMG1 = augmenting(box, ids, IMG1, transformed)
    if ids == 3:
        box = contourfind(black)
        IMG1 = augmenting(box, ids, IMG1, transformed)
    else:
        box = contourfind(peach)
        IMG1 = augmenting(box, ids, IMG1, transformed)

cv.imwrite("DONE.jpg", IMG1)
cv.imshow("Atulya", IMG1)
cv.waitKey(0)
cv.destroyAllWindows