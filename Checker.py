import cv2
import numpy as np

class Road:
    def __init__(self, obj) -> None:
        self.Image = obj
        self.Edge = obj#self.EdgeDetection(obj)
        _, self.Checking = cv2.threshold(self.Edge, 127, 255, cv2.THRESH_BINARY)
        # self.points = self.GetPoints(self.EdgeDetection(obj))
        # print(self.points)
        self.threshold = 10

    def EdgeDetection(self, Image):
    # Image = cv2.imread("Hand.png")
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, ksize = (5, 5), sigmaX= 0)
        
        canny = cv2.Canny(blur, 100, 200)
        return canny

    def GetPoints(self, Image):
        points = []
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        Image = cv2.threshold(Image, 127, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("Image", Image)
        for row in range(Image.shape[0]):
            for col in range(Image.shape[1]):
                intensity = Image[row][col]
                if intensity > 0:
                    points.append((col, row))
        return np.array(points)
    
    def dice_coefficient(self):
        y_true = cv2.imread("Edge.png", 0)
        y_pred = cv2.imread("Hand.png", 0)
        
        y_true = cv2.threshold(y_true, 127, 1, cv2.THRESH_BINARY)[1]
        y_pred = cv2.threshold(y_pred, 127, 1, cv2.THRESH_BINARY)[1]

        # Calculate the Dice coefficient
        intersection = np.sum(np.logical_and(y_true, y_pred))
        y_true_sum = np.sum(y_true)
        y_pred_sum = np.sum(y_pred)
        smooth = 1e-6
        dice = (2 * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
        # Return the Dice coefficient
        return dice
    
    def PreCanvas(self):
        edge = np.uint8(self.Edge)
        return edge
    
    def GetHighestPoint(self, points):
        Highest = points[0]
        for point in points:
            if point[1] < Highest[1]:
                Highest = point
        return Highest

    def Test(self, frame, points):
        for point in points:
            cv2.circle(frame, point, 25, (255, 255, 0))
        cv2.waitKey()
                

# Image = cv2.imread("Objects/circle.png")
# check = Road(Image)
# cv2.imshow("Edge", check.PreCanvas())
# cv2.waitKey()
# cv2.destroyAllWindows()