import cv2 as cv
import numpy as np
import math
import time

# Initialize video capture from file
cap = cv.VideoCapture("TennisBall/vid4Edit_Trim.mp4")
# cap = cv.VideoCapture(0)

prevCircle = None

# Lambda function to calculate the squared distance between two points
dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

x = [-99, -99, -99]
y = [-99, -99, -99]

frameCounter = 0
font = cv.FONT_HERSHEY_PLAIN

a = 0
b = 0
c = 0

# color range for yellow in HSV
yellow_lower = np.array([20, 80, 80])
yellow_upper = np.array([32, 255, 255])
trackWidth = 0

while cap.isOpened():
    ret, frame = cap.read()
    frameCounter += 1

    # Convert frame to grayscale and HSV apply Gaussian blur
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blurFrame = cv.GaussianBlur(grayFrame, (3, 3), 0)

    # Mask for yellow color
    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

    # Detect circles in the mask using HoughCircles method
    circles = cv.HoughCircles(mask_yellow, cv.HOUGH_GRADIENT, 1.5, 100, param1=5, param2=8, minRadius=5, maxRadius=20)

    if circles is not None:
        # Draw rectangle
        width, height = 900, 600
        img = np.zeros((height, width, 3), np.uint8)
        white_rect = np.array([[100, 100], [width - 100, 100], [width - 100, height - 100], [100, height - 100], [100, 100]], np.int32)
        img = cv.polylines(img, [white_rect], True, (255, 255, 255), 3)

        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i

        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)

        # Update tracking points
        if x[0] != -99 and x[1] != -99 and x[2] != -99 and y[0] != -99 and y[1] != -99 and y[2] != -99:
            if chosen is not None:
                x[0], x[1], x[2] = x[1], x[2], chosen[0]
                y[0], y[1], y[2] = y[1], y[2], chosen[1]
        else:
            if x[0] == -99:
                x[0], y[0] = chosen[0], chosen[1]
            elif x[1] == -99:
                x[1], y[1] = chosen[0], chosen[1]
            elif x[2] == -99:
                x[2], y[2] = chosen[0], chosen[1]

        cv.putText(frame, f'({chosen[0]},{chosen[1]})', (chosen[0], chosen[1]), font, 3, (100, 255, 0), 3, cv.LINE_8)

        # Draw tracking points
        cv.circle(frame, (x[0], y[0]), 6, (10, 255, 255), -1)
        cv.circle(frame, (x[1], y[1]), 6, (10, 255, 255), -1)

        # Fit a parabola if tracking points are valid
        if x[0] != -99 and x[1] != -99 and x[2] != -99 and y[0] != -99 and y[1] != -99 and y[2] != -99:
            b = (y[2] * (x[1] ** 2) - y[2] * (x[0] ** 2) - y[0] * (x[1] ** 2) - y[1] * (x[2] ** 2) + y[0] * (x[2] ** 2) + y[1] * (x[0] ** 2)) / ((x[0] - x[1]) * ((x[2] ** 2) + (x[0] * x[1]) - (x[0] * x[2]) - (x[1] * x[2])))
            a = (y[1] - y[0] + b * x[0] - b * x[1]) / ((x[1] ** 2) - (x[0] ** 2))
            c = -(x[0] ** 2) * a - b * x[0] + y[0]

            leftX = 1
            leftY = int(a * (leftX ** 2) + b * leftX + c)
            leftY = max(min(leftY, 2147483647), -2147483647)

            print(frameCounter)
            print(a)
            print(b)
            print(c)
            print("")

            pts = np.array([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [leftX, leftY]], np.int32)
            coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
            poly = np.poly1d(coeffs)

            print(pts)

            # Determine y range for polyline based on direction
            yarr = np.arange(-leftY, y[2]) if y[1] < y[0] else np.arange(y[2], leftY)
            xarr = poly(yarr)
            parab_pts = np.array([xarr, yarr], dtype=np.int32).T

            cv.polylines(frame, [parab_pts], False, (255, 0, 0), 3)

    # Display frame counter
    cv.putText(frame, str(frameCounter), (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)

    # Show frames
    cv.imshow('Video', frame)
    cv.imshow('Blurr', blurFrame)
    cv.imshow('HSV', hsv)
    cv.imshow('YellowMask', mask_yellow)

    # pause for playback speed
    cv.waitKey(1000)

    # Exit loop on spacebar press
    if cv.waitKey(1) & 0xFF == ord(' '):
        break

# Release resources and close windows
cap.release()
cv.destroyAllWindows()
