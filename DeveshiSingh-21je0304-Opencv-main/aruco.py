import cv2 as cv
import cv2.aruco as aruco

IMG = cv.imread('Downloads/CVtask.jpg')
IMG1 = cv.resize(IMG, (0, 0), fx=0.5, fy=0.5)  #reducing image size


def findarucomarker(img, markerSize=5, totalMarkers=250, draw=True):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucodict = aruco.Dictionary_get(
        key)  #dictionary to keep the aruco markers
    arucoparameter = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(
        gray, arucodict, parameters=arucoparameter
    )  #corners are the coordinates of the markers ids are the unique aruco identification for each 5x5 aruco
    print(ids)
    if len(corners) > 0:
        ids = ids.flatten()  #storing all the ids together

        for (markerCorner, markerId) in zip(corners, ids):
            corners = markerCorner.reshape(
                4, 2
            )  #4x2 matrix of all the coordiantes of the corners of the markers (x,y) for all 4 corners
            (top_left, top_right, bottom_right, bottom_left) = corners

            top_right = (int(top_right[0]), int(top_right[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

            cx = int(
                (top_left[0] + bottom_right[0]) / 2.0
            )  #midpoint of all markers irrespective of their orientation
            cy = int((top_left[1] + bottom_right[1]) / 2.0)


gray = cv.cvtColor(IMG1, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 240, 255,
                         cv.THRESH_BINARY)  #threshold values setting
contours, _ = cv.findContours(
    thresh, cv.RETR_TREE,
    cv.CHAIN_APPROX_SIMPLE)  #creating contours for shape detecting
for contour in contours:
    approx = cv.approxPolyDP(
        contour, 0.01 * cv.arcLength(contour, True), True
    )  #closed structure created by contours indicated by true, approx gives the coordinates of the corners of the contour lines
    cv.drawContours(IMG1, [approx], 0, (0, 0, 255), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx) == 4:  #four edges
        x, y, w, h = cv.boundingRect(approx)
        aspectRatio = float(w) / h  #since square has equal sides
        print(w, h)
        if (aspectRatio >= 0.97
                and aspectRatio <= 1.04):  #adjusting for margin of error
            cv.putText(IMG1, "Square", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5,
                       (0, 0, 0))

            print(approx)
        else:
            cv.putText(IMG1, "Quad", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5,
                       (0, 0, 0))

if __name__ == "__main__":
    img3 = cv.imread('Downloads/Ha.jpg')
    img1 = cv.imread('Downloads/LMAO.jpg')
    img4 = cv.imread('Downloads/HaHa.jpg')
    img2 = cv.imread('Downloads/XD.jpg')
    findarucomarker(img4)
    findarucomarker(img1)
    findarucomarker(img2)
    findarucomarker(img3)

    cornerblack = [[584, 38], [761, 37], [762, 214],
                   [585, 215]]  #coordinates of the square from the approx func
    cornerorange = [[702, 234], [841, 292], [783, 431], [644, 373]]
    cornergreen = [[100, 16], [301, 64], [253, 266], [51, 217]]
    cornerlilac = [[398, 208], [526, 442], [293, 569], [165, 336]]

    cornersid1 = [[50, 160], [450, 49], [561, 450], [161, 561]
                  ]  #coordinates of the bounding rectange of the aruco markers
    cornersid2 = [[122, 19], [575, 121], [473, 575], [19, 473]]
    cornersid3 = [[78, 77], [515, 77], [515, 515], [78, 515]]
    cornersid4 = [[28, 142], [450, 30], [562, 452], [140, 563]]

    img1_ = cv.resize(img1,
                      (251, 251))  #resizing the markers to fit the square
    img3_ = cv.resize(img3, (179, 179))
    img4_ = cv.resize(img4, (362, 362))
    img2_ = cv.resize(img2, (198, 198))
    h1, w1 = img1_.shape[:2]  #height and width of the markers
    h2, w2 = img2_.shape[:2]
    h3, w3 = img3_.shape[:2]
    h4, w4 = img4_.shape[:2]

    c1y, c1x = (h1 // 2, w1 // 2)
    c2y, c2x = (h2 // 2, w2 // 2)
    c3y, c3x = (h3 // 2, w3 // 2)
    c4y, c4x = (h4 // 2, w4 // 2)

    M1 = cv.getRotationMatrix2D(
        (c1x, c1y), -30, 1
    )  #rotation matrix to be applies and the angles to be rotated thorugh to fit the square boxes
    M2 = cv.getRotationMatrix2D(
        (c2x, c2y), 15, 1
    )  #calculated the angles manually, I couldn't figure out how else to do it
    M3 = cv.getRotationMatrix2D((c3x, c3y), -25, 1)
    M4 = cv.getRotationMatrix2D((c4x, c4y), 10, 1)

    rotated1 = cv.warpAffine(img1_, M1, (w1, h1))
    cv.imshow("1", rotated1)
    rotated2 = cv.warpAffine(
        img2_, M2, (w2, h2))  #warp function to rotate using the above matrices
    cv.imshow("2", rotated2)

    rotated3 = cv.warpAffine(img3_, M3, (w3, h3))
    cv.imshow("3", rotated3)

    rotated4 = cv.warpAffine(img4_, M4, (w4, h4))
    cv.imshow("4", rotated4)

    cv.imshow('img', IMG1)

    cv.waitKey(0)
    # couldn't figure out how to wrap the corner of the markers on the original image, I have tried many times unsuccessfully
cv.waitKey(0)
cv.destroyAllWindows