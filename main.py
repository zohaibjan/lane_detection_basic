import cv2
import matplotlib.pyplot as pyplot
import numpy as np
import sys


def convertToGray (image):
    gray = np.copy(image)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGBA2GRAY)
    return gray

def blurImage(grayImage):
    return cv2.GaussianBlur(grayImage, (5, 5), 0)

def detectEdges(blurImage):
    return cv2.Canny(blurImage, 50, 150)

def regionOfInterest(image):
    height = image.shape[0]
    #Single polygon in this case a Triangle that identifies the region of interest
    polygon = np.array([ [(200,height), (1100, height), (550, 250)] ])
    roi = np.zeros_like(image)
    cv2.fillPoly(roi, polygon, 255)
    mask = cv2.bitwise_and(image, roi)
    return mask

def drawLines(image, lines):
    drawnLines = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(drawnLines, (x1, y1), (x2, y2), (255,0,0), 10)
    return drawnLines

def avgLines(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = getCoordinates(image, left_fit_avg)
    right_line = getCoordinates(image, right_fit_avg)
    return np.array([left_line, right_line])

def getCoordinates(image, parameters):
    slope, intercept = parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1, x2,y2])

def mainProgram(image):
    laneImage = np.copy(image)
    im = convertToGray(image)
    im = blurImage(im)
    edges = detectEdges(im)
    croppedImage = regionOfInterest(edges)
    lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avg_lines = avgLines(laneImage, lines)
    drawnLines = drawLines(laneImage, avg_lines)
    mergedImage = cv2.addWeighted(laneImage, 0.8, drawnLines, 1, 1)
    cv2.imshow("Test Image", mergedImage)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        #im = cv2.imread("test_image.jpg")
        #laneImage = np.copy(im)
        file = sys.argv[1]
        flag = int(sys.argv[2])
        extension = file.split('.')[1]
        if flag == 1 and extension !='mp4':
            im = cv2.imread(file)
            mainProgram(im)
            cv2.waitKey(0)
        elif flag == 2:
            cap = cv2.VideoCapture(file)
            while cap.isOpened():
                _, frame = cap.read()
                mainProgram(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Please enter right parameters: main.py <FILENAME> <1/2>")
