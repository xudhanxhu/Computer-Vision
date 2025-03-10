import cv2 as cv 
import mediapipe as mp
import time
import os
import math

os.system('cls')

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelC= 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.tipIds = [4,8,12,16,20]

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 10, (255, 0, 255), cv.FILLED)

        return self.lmList
    

    def fingersUp(self, frame):
        fingers = []
        
        if len(self.lmList) != 0:
            #Thumb
            if((self.lmList[0][1] > frame.shape[1] // 2) and (self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1])):
                fingers.append(1)
            elif((self.lmList[0][1] > frame.shape[1] // 2) and (self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1])):
                fingers.append(1)
            else:
                fingers.append(0)
                
            #Rest
            for id in range(1, 5):
                if(self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]):
                    fingers.append(1)
                else:
                    fingers.append(0)
                
        return fingers
                
    def findDistance(self, frame, p1, p2):
        if len(self.lmList) != 0:
            x1, y1  = self.lmList[p1][1:]
            x2, y2  = self.lmList[p2][1:]
        
            return math.hypot(x2 - x1, y2 - y1)
        
        

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame = detector.findHands(frame)
        # lmList = detector.findPosition(frame)
        # if len(lmList) != 0:
        #     print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv.imshow("Image", frame)
        if cv.waitKey(1) & 0xFF==ord('x'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()