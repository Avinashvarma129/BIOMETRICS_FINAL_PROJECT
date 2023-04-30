import cv2
import mediapipe as mp
class Detect:
    def __init__(self):
        self.mface = mp.solutions.face_detection
        self.mppose = mp.solutions.pose
        self.mphands = mp.solutions.hands
        self.md = mp.solutions.drawing_utils
        self.ms= [self.mppose.PoseLandmark.LEFT_SHOULDER, self.mppose.PoseLandmark.RIGHT_SHOULDER]
        self.pose = self.mppose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mphands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5)
        self.g=None

    #Face
    def Face(self,frame):
        with self.mface.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face:
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            r1 = face.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if r1.detections:
                for detection in r1.detections:
                    location = detection.location_data
                    b = location.relative_bounding_box
                    rs = self.md._normalized_to_pixel_coordinates(b.xmin, b.ymin , w,h)
                    re = self.md._normalized_to_pixel_coordinates(b.xmin + b.width,b.ymin + b.height, w,h)
                    cv2.rectangle(frame, (rs[0],rs[1]-20),( re[0],re[1]-10),(255, 255, 255), 2)
        return frame       
    
    #Hand
    def Hands(self,gray,frame):
        r2 = self.hands.process(gray)
        if r2.multi_hand_landmarks:
            for hl in  (r2.multi_hand_landmarks):
                xList = []
                yList = []
                for lm in (hl.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    xList.append(px)
                    yList.append(py)
                X, xmax = min(xList), max(xList)
                Y, ymax = min(yList), max(yList)
                W, H = xmax - X, ymax - Y
                cv2.rectangle(frame, (X - 10, Y - 10),(X + W + 10, Y + H + 10),(0, 0,255), 2)
        return frame

    #Shoulder
    def Shoulder(self,gray,frame):
        r3 = self.pose.process(gray)
        ls=r3.pose_landmarks.landmark[self.ms[0]]
        rs=r3.pose_landmarks.landmark[self.ms[1]]
        if ls and rs:
            lx, ly = int(ls.x *w), int(ls.y *h)
            rx, ry = int(rs.x * w), int(rs.y * h)
            cv2.line(frame, (lx+20, ly-35), (rx-20, ry-35), (0, 255, 0), 3)
        if self.g is None:
            self.g=[ly,ry]
        #cv2.putText(frame, f"Global:{self.g[0],self.g[1]} Latest:{ly,ry}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #shrugg
    #here the threshold is set to the video as the distance between camera changes all the values change and the threshold needs to be adjusted.

        if (self.g[0]>ly or self.g[1]>ry) and (abs(self.g[0]-ly)>10 and abs(self.g[1]-ry)>10) and abs(ry - ly)<12:
            cv2.putText(frame, f"Shrugging Detected!", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame,f"No Shrugging Detected!", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        return frame
    
if __name__=="__main__":
    video = cv2.VideoCapture('FINAL.mp4')
    detectobj=Detect()
    while video.isOpened():
        r, frame = video.read()
        if r is None:
            break
        h,w,_= frame.shape
        frame=detectobj.Face(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame=detectobj.Hands(gray,frame)
        frame=detectobj.Shoulder(gray,frame)
        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    video.release()
    cv2.destroyAllWindows()