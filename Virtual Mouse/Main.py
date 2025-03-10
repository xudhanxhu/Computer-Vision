import cv2 as cv
import HandTrackModule as htm
import autopy as ap
import numpy as np
import time

cap = cv.VideoCapture(0)
detector = htm.handDetector(maxHands=1, detectionCon=0.7)

width, height = ap.screen.size()

frameReduct = 300
smooth = 7
currX, currY, preX, preY = 0, 0, 0, 0  # Initialize preX and preY

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def drawPointer(frame, lmList, index):
    x1, y1 = 0, 0
    if len(lmList) != 0:
        x1, y1 = lmList[index][1], lmList[index][2]
        x2, y2 = x1 + 5, y1 + 5

        cv.circle(frame, (x1, y1), 24, (245, 136, 72))
        cv.circle(frame, (x2, y2), 48, (240, 200, 177), 2)
    return x1, y1, frame

def drawFPS(frame):
    global pTime

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, f'FPS : {int(fps)}', (20, 40), cv.FONT_HERSHEY_PLAIN, 2, (235, 119, 52), 3)
    return frame

def getFingerDimension(lmList, index):
    x, y = 0, 0
    if len(lmList) != 0:
        x, y = lmList[index][1:]

    return x, y

def scroll_mouse(vertical_distance):
    for _ in range(abs(vertical_distance)):
        if vertical_distance > 0:
            ap.key.tap(ap.key.Code.UP_ARROW)
        else:
            ap.key.tap(ap.key.Code.DOWN_ARROW)
        time.sleep(0.01)

def mouseOps(lmList, fingers, frame):
    global currX, currY, preX, preY  # Declare preX and preY as global

    if len(fingers) != 0:
        hoverX, hoverY = getFingerDimension(lmList, 9)
        midX, midY = getFingerDimension(lmList, 12)

        if fingers[1] == 1 and fingers[2] == 1:
            _, _, frame = drawPointer(frame, lmList, 8)
            _, _, frame = drawPointer(frame, lmList, 12)

            hoverX = np.interp(hoverX, (frameReduct, frame.shape[1] - frameReduct), (0, width))
            hoverY = np.interp(hoverY, (frameReduct, frame.shape[0] - frameReduct), (0, height))

            if preX == 0 and preY == 0:  # Check if preX and preY are initialized
                preX, preY = midX, midY

            currX = preX + (hoverX - preX) / smooth
            currY = preY + (hoverY - preY) / smooth

            currX = np.clip(currX, 0, width)
            currY = np.clip(currY, 0, height)

            ap.mouse.move(currX, currY)
            preX, preY = currX, currY  # Assign values to preX and preY
            
        if fingers[1] == 0 and fingers[2] == 1:
            _, _, frame = drawPointer(frame, lmList, 12)

            ap.mouse.click()
            time.sleep(0.1) 

        if detector.findDistance(frame, 8, 12) < 60: # Add Scroll Here
            if fingers[1] == 1 and fingers[2] == 1:  # Check if both index and middle fingers are up
                # Calculate the vertical direction of the swipe
                vertical_direction = midY - hoverY
                #print(midY, hoverY)
                if vertical_direction < 0:  # Swipe from bottom to top
                    scroll_mouse(-1)  # Scroll up
                else:  # Swipe from top to bottom
                    scroll_mouse(1)                    

        if fingers[1] == 1 and fingers[2] == 0:
            _, _, frame = drawPointer(frame, lmList, 8)

            ap.mouse.click(ap.mouse.Button.RIGHT)

    return frame


pTime = 0

cv.namedWindow('Virtual Mouse', cv.WINDOW_NORMAL)
cv.setWindowProperty('Virtual Mouse', cv.WND_PROP_TOPMOST, 1)

while True:
    isTrue, frame = cap.read()

    frame = detector.findHands(rescaleFrame(cv.flip(frame, 1), 1.5))
    lmList = detector.findPosition(frame, draw=False)

    frame = drawFPS(frame)
    fingers = detector.fingersUp(frame)

    frame = mouseOps(lmList, fingers, frame)   
    
    cv.imshow('Virtual Mouse', frame)    

    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv.destroyAllWindows()
