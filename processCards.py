from re import S
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from glob import glob
from math import sqrt


def computeSlope(x: list, y: list):
    A = np.vstack([x, np.ones(len(x))]).T
    rtn = np.linalg.lstsq(A, y, rcond=None)
    m, c = rtn[0]
    err = sqrt(rtn[1][0]) if len(rtn[1]) else 0
    return m, err


def getDistance(p1: tuple, p2: tuple):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))


spread = 10
errThreshold = 4

for filename in sorted(glob('data/images/capture*.png')):
    # for filename in ['data/images/capture_023.png']:
    image = cv.imread(filename)

    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv.getRotationMatrix2D(center=center, angle=38, scale=1)
    image = cv.warpAffine(
        src=image, M=rotate_matrix, dsize=(width, height))
    orig = image.copy()

    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    # cv.imshow('frame', image)

    # Some cards have multiple contours, we're looking for the big one
    contour = sorted(contours, key=lambda a: len(a))[-1]
    # for point in contour:
    #     cv.drawMarker(image, point[0], (255, 0, 255))

    data = list()
    wasSide = None
    sideCount = 0
    index = len(contour)
    while index < 3*len(contour) and sideCount < 5:
        # for i in range(len(contour), 3*len(contour)):
        points = list(contour[a % len(contour)]
                      for a in range(index-spread, index+spread))
        x = list(a[0][0] for a in points)
        y = list(a[0][1] for a in points)

        m, err = computeSlope(x, y)
        # m2, err2 = computeSlope(y, x)
        isSide = (err < errThreshold)
        isSideStart = wasSide is not None and not wasSide and isSide
        if isSideStart:
            sideCount = sideCount + 1 if sideCount is not None else 0
        isSideEnd = wasSide is not None and wasSide and not isSide
        wasSide = isSide

        # if isSide:
        #     cv.drawMarker(image, (x[spread], y[spread]), (0, 255, 0))
        # image[y[spread], x[spread], :] = (255, 255, 0)
        # frame = image.copy()
        # cv.drawContours(frame, [np.array(points)], -1, (0, 0, 255), 3)
        # cv.imshow('frame', frame)
        data.append(
            [index, m, err, isSide, isSideStart, sideCount])
        print(data[-1])
        # cv.waitKey(0)
        index += spread if isSideEnd else 1

    print(filename)

    isSideList = list(a[3] for a in data)
    transitionsRaw = list(
        b and not a for a, b in zip(isSideList[:-1], isSideList[1:]))
    transitions = list(a for a, b in enumerate(transitionsRaw) if b)
    # assert(len(transitions) == 8)

    sidesData = list(a for a in data if a[3])
    x = list(a[0] for a in sidesData)
    m1 = list(a[1] for a in sidesData)
    e1 = list(a[2] for a in sidesData)
    sc = list(a[5]*100 for a in sidesData)

    # plt.plot(x, m1)
    # plt.scatter(x, e1)
    # plt.scatter(x, sc)
    # plt.show()

    sideCount = 0
    sideColorList = [
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    frame = image.copy()
    height, width, _ = frame.shape
    mlist, blist = list(), list()
    while sideCount < 4:
        sidePointIndices = list(
            a[0] for a in data if a[5] == (sideCount + 1) and a[3])
        i1, i2 = min(sidePointIndices)-spread+1, max(sidePointIndices)+spread
        points = list(contour[i % len(contour)] for i in range(i1, i2))

        x = list(a[0][0] for a in points)
        y = list(a[0][1] for a in points)
        x1, x2, y1, y2 = x[0], x[-1], y[0], y[-1]
        m = (y2-y1)/(x2-x1)
        b = y1 - x1*m
        mlist.append(m)
        blist.append(b)
        cv.line(frame, (0, int(b)), (width, int(width*m+b)), (0, 255, 0), 2)

        cv.drawContours(frame, [np.array(points)], -1,
                        sideColorList[sideCount], 3)
        sideCount += 1

    srcPoints = list()
    for i in range(4):
        i1, i2 = i, (i+1) % 4
        m1, b1, m2, b2 = mlist[i1], blist[i1], mlist[i2], blist[i2]
        # m1 * x + b1 = m2 * x + b2
        # x * (m1 - m2) = b2 - b1
        x = int((b2 - b1)/(m1 - m2))
        y = int(m1*x + b1)
        # cv.circle(frame, (x, y), 23, (255, 0, 255), 2)
        srcPoints.append((x, y))
        # cv.drawMarker(frame, (x, y), (255, 0, 255))

    tl = sorted(srcPoints, key=lambda a: a[0])[0]
    cv.circle(frame, tl, 23, (255, 0, 0), 2)
    tr = sorted(srcPoints, key=lambda a: a[1])[0]
    cv.circle(frame, tr, 23, (0, 255, 0), 2)
    bl = sorted(srcPoints, key=lambda a: a[1])[-1]
    cv.circle(frame, bl, 23, (0, 0, 255), 2)

    d1 = getDistance(tl, tr)
    d2 = getDistance(tl, bl)

    pts1 = np.float32([tl, tr, bl])
    pts2 = np.float32([(0, 0), (int(d1), 0), (0, int(d2))])
    rows, cols, ch = image.shape
    M = cv.getAffineTransform(pts1, pts2)
    trans = cv.warpAffine(orig, M, (cols, rows))
    trans = trans[:int(d2), :int(d1)]

    cv.imshow('frame', frame)
    cv.imshow('trans', trans)

    # trans = image.copy()

    cv.waitKey(0)


# walk the contour and find 4 common slopes to compute the lines
# Line shoudl get more accurate the further the two points are from each other
# Corners should make a mess and that's totally fine
# use the 4 lines ot find the corners we'll then build our affline transform

# cv.imshow('frame', image)
# cv.waitKey(0)
