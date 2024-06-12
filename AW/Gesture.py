# Imports

import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum

from google.protobuf.json_format import MessageToDict

import numpy as np
pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import computer vision packages
import cv2
import imutils
from imutils.video import VideoStream

# import keras packages
import keras
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json

# import statistic/data packages
from collections import deque
import numpy as np

# import utility packages
import argparse
import time
import os

characters = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z', 36:'a', 37:'b', 38:'c', 39:'d',
40:'e', 41:'f', 42:'g', 43:'h', 44:'i', 45:'j', 46:'k', 47:'l', 48:'m', 49:'n',
50:'o', 51:'p', 52:'q', 53:'r', 54:'s', 55:'t', 56:'u', 57:'v', 58:'w', 59:'x',
60:'y', 61:'z'}

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
#ap.add_argument("-b", "--buffer", type=int, default=64,
#    help="max buffer size")
#ap.add_argument("-vb", "--verbose", help="increase output verbosity", 
#    action="store_true")
args = vars(ap.parse_args())

# load keras model
def load_model():
    # Load trained model
    #if args.verbose:
    #    # print("Loading cnn model from disk.............", end="")

    # Load JSON model
    json_file = open(r'E:\B TECH PROJECTS2023\VJEC\AirWriting\AW\model_saves/cnn_model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # Load model weights
    model.load_weights(r"E:\B TECH PROJECTS2023\VJEC\AirWriting\AW\model_saves/cnn_model_weights.h5")
    #if args.verbose:
    #    # print("...finished.")
    return model

# predict letter given a model and image
def predict_model(model, image):
    prediction = model.predict(image.reshape(1,28,28,1))[0]
    prediction = np.argmax(prediction)
    return prediction



# used in calculating depth of object from camera
def update_depth(obj_width, focal_len, width):
    obj_depth = 0
    if width == 0:
        return obj_depth, 0
    elif obj_width == 0 or focal_len == 0:
        return obj_depth, 0
    else:
        obj_depth = obj_width * focal_len / width
        ## print("Width: ", width)
        ## print("Calculating depth...")
        ## print("Depth: ", obj_depth, "  delta: ", obj_depth-focal_len)
    return obj_depth, obj_depth-focal_len




class Gest(IntEnum):
    # Binary Encoded
    """
    Enum for mapping all hand gesture to binary number.
    """

    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    
    # Extra Mappings
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    """
    Convert Mediapipe Landmarks to recognizable Gestures.
    """
    
    def __init__(self, hand_label):
        """
        Constructs all the necessary attributes for the HandRecog object.

        Parameters
        ----------
            finger : int
                Represent gesture corresponding to Enum 'Gest',
                stores computed gesture for current frame.
            ori_gesture : int
                Represent gesture corresponding to Enum 'Gest',
                stores gesture being used.
            prev_gesture : int
                Represent gesture corresponding to Enum 'Gest',
                stores gesture computed for previous frame.
            frame_count : int
                total no. of frames since 'ori_gesture' is updated.
            hand_result : Object
                Landmarks obtained from mediapipe.
            hand_label : int
                Represents multi-handedness corresponding to Enum 'HLabel'.
        """

        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
    
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        """
        returns signed euclidean distance between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign
    
    def get_dist(self, point):
        """
        returns euclidean distance between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
    
    def get_dz(self,point):
        """
        returns absolute difference on z-axis between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    # Function to find Gesture Encoding using current finger_state.
    # Finger_state: 1 if finger is open, else 0
    def set_finger_state(self):
        """
        set 'finger' by computing ratio of distance between finger tip 
        , middle knuckle, base knuckle.

        Returns
        -------
        None
        """
        if self.hand_result == None:
            return

        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0 #thumb
        for idx,point in enumerate(points):
            
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            
            try:
                ratio = round(dist/dist2,1)
            except:
                ratio = round(dist1/0.01,1)

            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1
    

    # Handling Fluctations due to noise
    def get_gesture(self):


        if self.hand_result == None:
            return Gest.PALM

        current_gesture = Gest.PALM
        if self.finger in [Gest.LAST3,Gest.LAST4] and self.get_dist([8,4]) < 0.05:
            if self.hand_label == HLabel.MINOR :
                current_gesture = Gest.PINCH_MINOR
            else:
                current_gesture = Gest.PINCH_MAJOR

        elif Gest.FIRST2 == self.finger :
            point = [[8,12],[5,9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1/dist2
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8,12]) < 0.1:
                    current_gesture =  Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture =  Gest.MID
            
        else:
            current_gesture =  self.finger
        
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4 :
            self.ori_gesture = current_gesture


        return self.ori_gesture
# Executes commands according to detected gestures
class Controller:
    """
    Executes commands according to detected gestures.

    Attributes
    ----------
    tx_old : int
        previous mouse location x coordinate
    ty_old : int
        previous mouse location y coordinate
    flag : bool
        true if V gesture is detected
    grabflag : bool
        true if FIST gesture is detected
    pinchmajorflag : bool
        true if PINCH gesture is detected through MAJOR hand,
        on x-axis 'Controller.changesystembrightness', 
        on y-axis 'Controller.changesystemvolume'.
    pinchminorflag : bool
        true if PINCH gesture is detected through MINOR hand,
        on x-axis 'Controller.scrollHorizontal', 
        on y-axis 'Controller.scrollVertical'.
    pinchstartxcoord : int
        x coordinate of hand landmark when pinch gesture is started.
    pinchstartycoord : int
        y coordinate of hand landmark when pinch gesture is started.
    pinchdirectionflag : bool
        true if pinch gesture movment is along x-axis,
        otherwise false
    prevpinchlv : int
        stores quantized magnitued of prev pinch gesture displacment, from 
        starting position
    pinchlv : int
        stores quantized magnitued of pinch gesture displacment, from 
        starting position
    framecount : int
        stores no. of frames since 'pinchlv' is updated.
    prev_hand : tuple
        stores (x, y) coordinates of hand in previous frame.
    pinch_threshold : float
        step size for quantization of 'pinchlv'.
    """

    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    pinch_threshold = 0.3
    
    def getpinchylv(hand_result):
        """returns distance beween starting pinch y coord and current hand position y coord."""
        dist = round((Controller.pinchstartycoord - hand_result.landmark[8].y)*10,1)
        return dist

    def getpinchxlv(hand_result):
        """returns distance beween starting pinch x coord and current hand position x coord."""
        dist = round((hand_result.landmark[8].x - Controller.pinchstartxcoord)*10,1)
        return dist
    
    # def changesystembrightness():
    #     """sets system brightness based on 'Controller.pinchlv'."""
    #     currentBrightnessLv = sbcontrol.get_brightness(display=0)/100.0
    #     currentBrightnessLv += Controller.pinchlv/50.0
    #     if currentBrightnessLv > 1.0:
    #         currentBrightnessLv = 1.0
    #     elif currentBrightnessLv < 0.0:
    #         currentBrightnessLv = 0.0
    #     sbcontrol.fade_brightness(int(100*currentBrightnessLv) , start = sbcontrol.get_brightness(display=0))
    #
    # def changesystemvolume():
    #     """sets system volume based on 'Controller.pinchlv'."""
    #     devices = AudioUtilities.GetSpeakers()
    #     interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    #     volume = cast(interface, POINTER(IAudioEndpointVolume))
    #     currentVolumeLv = volume.GetMasterVolumeLevelScalar()
    #     currentVolumeLv += Controller.pinchlv/50.0
    #     if currentVolumeLv > 1.0:
    #         currentVolumeLv = 1.0
    #     elif currentVolumeLv < 0.0:
    #         currentVolumeLv = 0.0
    #     volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)
    
    # def scrollVertical():
    #     """scrolls on screen vertically."""
    #     pyautogui.scroll(120 if Controller.pinchlv>0.0 else -120)
    #
    #
    # def scrollHorizontal():
    #     """scrolls on screen horizontally."""
    #     pyautogui.keyDown('shift')
    #     pyautogui.keyDown('ctrl')
    #     pyautogui.scroll(-120 if Controller.pinchlv>0.0 else 120)
    #     pyautogui.keyUp('ctrl')
    #     pyautogui.keyUp('shift')

    # Locate Hand to get Cursor Position
    # Stabilize cursor by Dampening
    def get_position(hand_result):
        """
        returns coordinates of current hand position.

        Locates hand to get cursor position also stabilize cursor by 
        dampening jerky motion of hand.

        Returns
        -------
        tuple(float, float)
        """
        point = 9
        position = [hand_result.landmark[point].x ,hand_result.landmark[point].y]
        sx,sy = pyautogui.size()
        x_old,y_old = pyautogui.position()
        x = int(position[0]*sx)
        y = int(position[1]*sy)
        if Controller.prev_hand is None:
            Controller.prev_hand = x,y
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]

        distsq = delta_x**2 + delta_y**2
        ratio = 1
        Controller.prev_hand = [x,y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1/2))
        else:
            ratio = 2.1
        x , y = x_old + delta_x*ratio , y_old + delta_y*ratio
        return (x,y)

    def pinch_control_init(hand_result):
        """Initializes attributes for pinch gesture."""
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv = 0
        Controller.prevpinchlv = 0
        Controller.framecount = 0

    # Hold final position for 5 frames to change status
    def pinch_control(hand_result, controlHorizontal, controlVertical):
        """
        calls 'controlHorizontal' or 'controlVertical' based on pinch flags, 
        'framecount' and sets 'pinchlv'.

        Parameters
        ----------
        hand_result : Object
            Landmarks obtained from mediapipe.
        controlHorizontal : callback function assosiated with horizontal
            pinch gesture.
        controlVertical : callback function assosiated with vertical
            pinch gesture. 
        
        Returns
        -------
        None
        """
        if Controller.framecount == 5:
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv

            if Controller.pinchdirectionflag == True:
                controlHorizontal() #x

            elif Controller.pinchdirectionflag == False:
                controlVertical() #y

        lvx =  Controller.getpinchxlv(hand_result)
        lvy =  Controller.getpinchylv(hand_result)
            
        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0

        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    def handle_controls(gesture, hand_result):  
        """Impliments all gesture functionality."""      
        x,y = None,None
        if gesture != Gest.PALM :
            x,y = Controller.get_position(hand_result)
        
        # flag reset
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button = "left")

        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        # implementation
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration = 0.1)

        elif gesture == Gest.FIST:
            if not Controller.grabflag : 
                Controller.grabflag = True
                pyautogui.mouseDown(button = "left")
            pyautogui.moveTo(x, y, duration = 0.1)

        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False

        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button='right')
            Controller.flag = False

        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False

        elif gesture == Gest.PINCH_MINOR:
            if Controller.pinchminorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(hand_result,Controller.scrollHorizontal, Controller.scrollVertical)
        
        elif gesture == Gest.PINCH_MAJOR:
            if Controller.pinchmajorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(hand_result,Controller.changesystembrightness, Controller.changesystemvolume)

'''
----------------------------------------  Main Class  ----------------------------------------
    Entry point of Gesture Controller
'''


class GestureController:
    """
    Handles camera, obtain landmarks from mediapipe, entry point
    for whole program.

    Attributes
    ----------
    gc_mode : int
        indicates weather gesture controller is running or not,
        1 if running, otherwise 0.
    cap : Object
        object obtained from cv2, for capturing video frame.
    CAM_HEIGHT : int
        highet in pixels of obtained frame from camera.
    CAM_WIDTH : int
        width in pixels of obtained frame from camera.
    hr_major : Object of 'HandRecog'
        object representing major hand.
    hr_minor : Object of 'HandRecog'
        object representing minor hand.
    dom_hand : bool
        True if right hand is domaniant hand, otherwise False.
        default True.
    """
    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None # Right Hand by default
    hr_minor = None # Left hand by default
    dom_hand = True

    def __init__(self):
        """Initilaizes attributes."""
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    def classify_hands(results):
        """
        sets 'hr_major', 'hr_minor' based on classification(left, right) of 
        hand obtained from mediapipe, uses 'dom_hand' to decide major and
        minor hand.
        """
        left , right = None,None
        try:
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else :
                left = results.multi_hand_landmarks[0]
        except Exception as e:
            # print(e)

            pass

        try:
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else :
                left = results.multi_hand_landmarks[1]
        except:
            pass
        
        if GestureController.dom_hand == True:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else :
            GestureController.hr_major = left
            GestureController.hr_minor = right

    def start(self):
        """
        Entry point of whole programm, caputres video frame and passes, obtains
        landmark from mediapipe and passes it to 'handmajor' and 'handminor' for
        controlling.
        """
        
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)
        currentgustflag = ""
        pixellist=[]
        model = load_model()
        output=""
        cv=0
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, image = GestureController.cap.read()

                if not success:
                    # # print("Ignoring empty camera frame.")
                    continue
              
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)

                image.flags.writeable = True
                size=image.shape

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                rs=""
                if results.multi_hand_landmarks:

                    handLandmarks = []

                    # Fill list with x and y positions of each landmark

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get hand index to check label (left or right)
                        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                        handLabel = results.multi_handedness[handIndex].classification[0].label

                        # Set variable to keep landmarks positions (x and y)
                        handLandmarks = []

                        # Fill list with x and y positions of each landmark
                        # # print(len( hand_landmarks.landmark))
                        for landmarks in hand_landmarks.landmark:
                            handLandmarks.append([landmarks.x, landmarks.y])

                        break



                    GestureController.classify_hands(results)
                    handmajor.update_hand_result(GestureController.hr_major)
                    handminor.update_hand_result(GestureController.hr_minor)

                    handmajor.set_finger_state()
                    handminor.set_finger_state()
                    gest_name,points = handminor.get_gesture(),""


                    if gest_name == Gest.PINCH_MINOR:
                        # print(gest_name,"+=+=+=+=+=",currentgustflag)
                        if currentgustflag != gest_name:
                            pass
                            # print("========================================")



                        # Controller.handle_controls(gest_name, handminor.hand_result)
                    else:
                        gest_name,points = handmajor.get_gesture(),""
                        print(gest_name,"_+_+_+_+_+_+_+",currentgustflag)
                        if str(gest_name)=="15":
                            # print("*****************************************")
                            if str(currentgustflag)!=str(gest_name):
                                pass
                                # print("+++++++++++++++++++++++++++++++++++++++++++++++++")


                        if str(gest_name)=="8":

                            # print("+++++++++++++++++++++++++++++++++")
                            # print("+++++++++++++++++++++++++++++++++")
                            # print("+++++++++++++++++++++++++++++++++")
                            # print(handLandmarks[8])
                            x=size[1]*handLandmarks[8][0]
                            y=size[0]*handLandmarks[8][1]
                            # print(x,y)
                            pixellist.append((int(y),int(x)))

                            # print("*****************************************")
                            if str(currentgustflag)!=str(gest_name):
                                pass
                                # print("+++++++++++++++++++++++++++++++++++++++++++++++++")
                        else:
                            try:
                                if pixellist[-1]!=(None,None):
                                    pixellist.append((None,None))
                            except:
                                pass

                        if str(gest_name) == "Gest.V_GEST":

                            if currentgustflag!=gest_name:
                                print(str(gest_name)), "======================"
                                mirrored_board = blackboard.copy()
                                blackboard_gray = cv2.cvtColor(mirrored_board, cv2.COLOR_BGR2GRAY)
                                # cv2.imwrite("blackboard_gray_image.png", blackboard_gray)
                                blur1 = cv2.medianBlur(blackboard_gray, 15)
                                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                                # cv2.imwrite("blurredimage.png", blur1)
                                ret, thresh = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                # Finding contours on the blackboard
                                blackboard_cnts, blackboard_hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                                                                    cv2.CHAIN_APPROX_NONE)
                                # blackboard_img, blackboard_cnts, blackboard_hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                # cv2.imwrite("contourblackboard.png", blackboard_img)

                                if len(blackboard_cnts) > 0:
                                    cnt = sorted(blackboard_cnts, key=cv2.contourArea, reverse=True)[0]
                                    if cv2.contourArea(cnt) > 1000:
                                        x, y, w, h = cv2.boundingRect(cnt)
                                        alphabet = blackboard_gray[y - 10:y + h + 10, x - 10:x + w + 10]
                                        newImage = cv2.resize(alphabet, (28, 28))

                                        # predict char digit
                                        r = characters[predict_model(model, newImage)]
                                        print(r)
                                        output += r
                                        print(output)
                                        pixellist = []
                                        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

                        if str(gest_name) == "14":
                            if str(gest_name)!=currentgustflag:
                                if len(output)!=0:
                                    output=output[:-1]
                                print(output)
                        if str(gest_name) == "1":
                            GestureController.cap.release()
                            cv2.destroyAllWindows()
                            return output
                        currentgustflag=str(gest_name)

                    # for hand_landmarks in results.multi_hand_landmarks:
                    #     # print(type(hand_landmarks),"+++++++++++++++++++")
                    #     # print("@@@@@@@@@@@@@@@@@@@")
                    #
                    #     index_finger_tip_index = 8
                    #
                    #     # Extract the index finger tip landmark
                    #     import mediapipe as mp
                    #     try:
                    #         landmark_list = mp.solutions.hands.Hands().process(image).multi_hand_landmarks[0]
                    #
                    #         # Define the index of the index finger tip landmark (usually 8 for the index finger tip)
                    #         index_finger_tip_index = 8
                    #
                    #         # Extract the index finger tip landmark
                    #         index_finger_tip_landmark = landmark_list.landmark[index_finger_tip_index]
                    #
                    #         # Extract the x and y coordinates of the index finger tip
                    #         x = index_finger_tip_landmark.x
                    #         y = index_finger_tip_landmark.y
                    #         # print(x,y,"@@@@@@@@@@@")
                    #     except:
                    #         pass
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    Controller.prev_hand = None
                f=0
                pv=0
                for i in pixellist:
                    if f==1:
                        if i[1]!=None and pv[0]!=None:
                            cv2.line(image, pv, (i[1],i[0]), (255, 255, 255), 2)
                            try:
                                cv2.line(blackboard,pv, (i[1],i[0]), (255, 255, 255), 8)
                            except:
                                pass
                    # image[i[0],i[1]]=(255,0,0)
                    f=1

                    pv=(i[1],i[0])
                cv2.imshow('Video Frame', image)
                mirrored_board = blackboard.copy()
                # mirrored_board = cv2.flip(blackboard, 1)
                cv2.imshow("Blackboard", mirrored_board)
                key = cv2.waitKey(1) & 0xFF
                # if key == ord("d"):
                #     # cv2.imwrite("blackboardimage.png", mirrored_board)
                #     blackboard_gray = cv2.cvtColor(mirrored_board, cv2.COLOR_BGR2GRAY)
                #     # cv2.imwrite("blackboard_gray_image.png", blackboard_gray)
                #     blur1 = cv2.medianBlur(blackboard_gray, 15)
                #     blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                #     # cv2.imwrite("blurredimage.png", blur1)
                #     ret, thresh = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                #     # Finding contours on the blackboard
                #     blackboard_cnts, blackboard_hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                #                                                         cv2.CHAIN_APPROX_NONE)
                #     # blackboard_img, blackboard_cnts, blackboard_hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                #     # cv2.imwrite("contourblackboard.png", blackboard_img)
                #
                #     if len(blackboard_cnts) > 0:
                #         cnt = sorted(blackboard_cnts, key=cv2.contourArea, reverse=True)[0]
                #         if cv2.contourArea(cnt) > 1000:
                #             x, y, w, h = cv2.boundingRect(cnt)
                #             alphabet = blackboard_gray[y - 10:y + h + 10, x - 10:x + w + 10]
                #             newImage = cv2.resize(alphabet, (28, 28))
                #
                #             # predict char digit
                #             r=characters[predict_model(model, newImage)]
                #             print(r)
                #             output+=r
                #             print(output)
                #
                #             # # save image to disk
                #             # path = 'input_images/'
                #             # cv2.imwrite(os.path.join(path, "img-%d.png" % count), newImage)
                #
                #     pixellist=[]
                #     blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                #     pts = deque(maxlen=512)
                if cv2.waitKey(5) & 0xFF == 13:
                    break
        GestureController.cap.release()
        cv2.destroyAllWindows()

# uncomment to run directly
# gc1 = GestureController()
# gc1.start()
