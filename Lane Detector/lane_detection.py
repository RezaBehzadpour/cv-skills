import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

class LaneDetection:
    def __init__(self, roi):
        self.roi = roi

    def make_points(self, image, line):
        """
        Create points in the image frame
        """
        slope, intercept = line
        y1 = int(image.shape[0])   # bottom of the image
        y2 = int(y1 * 3/5)         # slightly lower than the middle
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    def average_slope_intercept(self, image, lines):
        """
        Use only a single line which is an average of close lines
        """
        left_fit    = []
        right_fit   = []
        if lines is None:
            return None
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1,x2), (y1,y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0: # y is reversed in image
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
        # add more weight to longer lines
        left_fit_average  = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line  = self.make_points(image, left_fit_average)
        right_line = self.make_points(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines

    def canny(self, image):
        """
        Find edges in the grayscale image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = 5
        blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def display_lines(self, image, lines):
        """
        Draw the lines
        """
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255), 9)
        return line_image

    def region_of_interest(self, canny):
        """
        Only focus the region of the road
        """
        mask = np.zeros_like(canny)
        cv2.fillPoly(mask, self.roi, 255)
        masked_image = cv2.bitwise_and(canny, mask)
        return masked_image

dir = 'content'
image_name = os.path.join(dir, 'image1.jpg')
video_name = os.path.join(dir, 'video1.mp4')

image = cv2.imread(image_name)
image = cv2.resize(image, (1000, 500))
lane_image = np.copy(image)
height = lane_image.shape[0]
width = lane_image.shape[1]
roi = np.array([[(width / 4.5, height),
                (width / 2 , height / 2), 
                (width, height)]], np.int32)

lane = LaneDetection(roi)

cap = cv2.VideoCapture(video_name)
while(cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        break
    lane_image = np.copy(frame)
    canny_image = lane.canny(lane_image)
    cropped_canny = lane.region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 3, np.pi/180, 135, np.array([]), minLineLength=30, maxLineGap=6)
    averaged_lines = lane.average_slope_intercept(lane_image, lines)
    line_image = lane.display_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.imshow('Result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
