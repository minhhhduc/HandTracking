import cv2
import mediapipe as mp
import time
import numpy as np
import os, random
from Checker import Road
import numpy as np
 
class handDetector():
    def __init__(self, mode=False, maxHands=1, modelC = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelC = modelC
        self.mpHands = mp.solutions.hands
        self.indexFinger = 4
        # self.hands = self.mpHands.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                                self.mpHands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, handNo=0, draw=True):
 
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
 
        return lmList

def CheckDraw(lmList, StartPoint, radius, CanStart):
    StartPoint_np = np.array(StartPoint)
    lmList_np = np.array(lmList[8][1:])
    return  np.linalg.norm(StartPoint_np - lmList_np) > radius and CanStart

def CanStart(lmList, StartPoint, radius):
    StartPoint_np = np.array(StartPoint)
    lmList_np = np.array(lmList[8][1:])
    return np.linalg.norm(StartPoint_np - lmList_np) <= radius

def GetObjectToShow():
    Path = "Objects/"
    files = os.listdir(Path)
    Object = random.choice(files)
    # Object = "Triangle.png"
    return cv2.imread(Path + Object)

def draw_accurency(frame, dice):
    # font
    dice = round(dice, 3) * 100
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (frame.shape[1] - 100 , 70)
    
    # fontScale
    fontScale = 1
    
    # Red color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.putText() method
    image = cv2.putText(frame, str(dice) + '%', org, font, fontScale, 
                    color, thickness, cv2.LINE_AA, False)
    return image

def main():
    pTime = 0
    cTime = 0
    Object = GetObjectToShow()
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    points = []
    Roads = Road(Object)
    StartPointRadius = 25
    StartPoint = (300, 100)
    canStart = False
    canStop = False
    canDraw = False
    print(Roads.Image.shape)
    # cv2.imshow("Target", Object)
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img, draw= True)
        lmList = detector.findPosition(img, draw= False)
        # print(canStart)
        if len(lmList) > 1 and StartPoint != (0, 0) :
            if canStart == False and canStop == False:
                canStart = CanStart(lmList, StartPoint, StartPointRadius)
            elif CanStart(lmList, StartPoint, StartPointRadius) and canDraw == True:
                canStart = False
                canStop = True
            
            if canStart and CanStart(lmList, StartPoint, StartPointRadius) == False:
                canDraw = True
            
            cv2.circle(img, lmList[8][1:], 15, color = (187, 181,255), thickness= cv2.FILLED)
            # cv2.circle(img, lmList[6][1:], 15, color = (0, 0,255), thickness= cv2.FILLED) #This checker
            if CheckDraw(lmList, StartPoint, StartPointRadius, canStart):
                points.append(lmList[8][1:])
        #Draw Color2
        CanvasImage = np.zeros(img.shape, dtype= np.uint8)
        RoadImage = Roads.PreCanvas()
        offsetx = 150
        offsety = 50
        CanvasImage[offsety: offsety + RoadImage.shape[0], offsetx : offsetx + RoadImage.shape[1]] = RoadImage
        NewCanvas = np.zeros(CanvasImage.shape)
        # CanvasImage = cv2.addWeighted(CanvasImage, 1, 
        # RoadImage, 0.2, 0)
        Origin = CanvasImage
        cv2.imwrite("Edge.png", Origin)
        if StartPoint == "":
            StartPoint = Roads.GetHighestPoint(Roads.GetPoints(Origin))
        for ptIdx in range(len(points) - 1):
            startpoint = points[ptIdx]
            endpoint = points[ptIdx + 1]
            cv2.line(CanvasImage, startpoint, endpoint, color=(187, 181,255), thickness= 18)
            cv2.line(NewCanvas, startpoint, endpoint, color=(255, 255, 255), thickness= 18)

        cv2.imwrite("Hand.png", NewCanvas)
        img = cv2.add(img, CanvasImage)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.circle(img, StartPoint, StartPointRadius, (255, 255, 0), thickness=cv2.FILLED)
        
        #Check if go collect side
        
        # print(Roads.dice_coefficient())
        draw_accurency(img, Roads.dice_coefficient())

        cv2.imshow("Image", img)
        cv2.imshow("Canvas", Origin)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('w'):
            points = []
            canStart = False
            canStop = False
            canDraw = False
            
        # if key == ord('e'):
        #     CheckMSE(Object, CanvasImage)



if __name__ == "__main__":
    main()