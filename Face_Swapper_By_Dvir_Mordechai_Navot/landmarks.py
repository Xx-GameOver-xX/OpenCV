import cv2
import dlib 
import numpy
import sys

picPath = sys.argv[1]

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class NoFaces(Exception):
    pass

class ToManyFaces(Exception):
    pass
    
def Finde_Landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise ToManyFaces
    if len(rects) == 0:
        raise NoFaces
        
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0,255,255))
    return im

image = cv2.imread(str(picPath))
landmarks = Finde_Landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)

cv2.imshow('Result', image_with_landmarks)
cv2.imwrite('image_with_landmarks.jpg', image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()