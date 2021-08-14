import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax, argmin
from numpy.core.shape_base import hstack, vstack
import cv2


h_lower = 100
def set_h_lower(x):
    global h_lower
    h_lower = x
s_lower = 70
def set_s_lower(x):
    global s_lower
    s_lower = x
v_lower = 80
def set_v_lower(x):
    global v_lower
    v_lower = x

h_upper = 175
def set_h_upper(x):
    global h_upper
    h_upper = x
s_upper = 255
def set_s_upper(x):
    global s_upper
    s_upper = x
v_upper = 255
def set_v_upper(x):
    global v_upper
    v_upper = x


def dist(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    return (a * a + b * b) ** 0.5

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	max_width = max(int(width_a), int(width_b))

	height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	max_height = max(int(height_a), int(height_b))

	dst = np.array([
	    [0, 0],
	    [max_width - 1, 0],
	    [max_width - 1, max_height - 1],
	    [0, max_height - 1]],
                       dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (max_width, max_height))

	return warped



def clip_approx_rect(img: np.ndarray):
    img = img.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, (h_lower, s_lower, v_lower), (h_upper, s_upper, v_upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        hull = cv2.convexHull(contour)
        length = cv2.arcLength(hull, True)
        dp = cv2.approxPolyDP(hull, 0.02 * length, True)
        points = cv2.boxPoints(cv2.minAreaRect(contour))
        points = np.int0(points)
        return four_point_transform(img, points)
        # cv2.drawContours(img, [points], 0, (255, 0, 0), 3)
    
    return None

def read_templates():
    templates = []
    for i in range(10):
        img = cv2.imread('templates/{}.png'.format(i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]
        templates.append(img)
    return templates

templates = read_templates()

def detect(img_in):
    h, w, _ = img_in.shape
    if w / h < 0.5:
        return img_in
    result = cv2.resize(img_in, (int(144 * w / h), 144))
    img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    rect_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    rect_img = cv2.dilate(rect_img, (5, 5))
    rect_contours, _ = cv2.findContours(rect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for rect_contour in rect_contours:
        rx, ry, rw, rh = cv2.boundingRect(rect_contour)
        rx = int(rx + rw - rh / 1.44)
        rw = int(rh / 1.44)
        
        ry -= 4
        rx -= 2
        rw += 5
        rh += 4
        rects.append((rx, ry, rx + rw, ry + rh))
        cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
    # cv2.imshow('Test', hstack([org, cv2.cvtColor(rect_img, cv2.COLOR_GRAY2BGR)]))
    # cv2.waitKey(0)
    
    for fx, fy, tx, ty in rects:
        roi = img[max(0, fy - 5): min(ty + 5, h), max(0, fx - 5):min(tx + 5, w)]
        # cv2.imshow('ROI', roi)
        # cv2.waitKey(0)
        scores = []
        for template in templates:
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
               break
            match_result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(match_result)
            scores.append(score)
        if len(scores) > 0:
            cv2.putText(result, str(argmax(scores)), (fx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return result
    # cv2.imshow('Result', org)
    # cv2.waitKey(0)
    
def main():
    WINNAME = "Camera"
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINNAME)
    def trackBar(*names):
        return [cv2.getTrackbarPos(name, WINNAME) for name in names]
    cv2.createTrackbar("H Lower", WINNAME, h_lower, 180, set_h_lower)
    cv2.createTrackbar("S Lower", WINNAME, s_lower, 255, set_s_lower)
    cv2.createTrackbar("V Lower", WINNAME, v_lower, 255, set_v_lower)
    cv2.createTrackbar("H Upper", WINNAME, h_upper, 180, set_h_upper)
    cv2.createTrackbar("S Upper", WINNAME, s_upper, 255, set_s_upper)
    cv2.createTrackbar("V Upper", WINNAME, v_upper, 255, set_v_upper)
    
    while cap.isOpened():
        _, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img_crop = clip_approx_rect(img)
        # img = clipApproxRect(img)
        if img_crop is not None:
            cv2.imshow(WINNAME, detect(img_crop))
        if cv2.waitKey(15) > 0:
            break
    
main()
