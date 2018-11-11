import dlib
import cv2
import numpy
import numpy as np
from time import sleep 
import sys

firstPicPath = sys.argv[1]
secondPicPath = sys.argv[2]

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

# face points list (range)
FACE_POINTS = list(range(17, 68))

# eyes points (ranges)
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

# eyebrows points (ranges)
LEFT_EYEBROW_POINTS = list(range(22, 27))
RIGHT_EYEBROW_POINTS = list(range(17, 22))

# nose points list (range)
NOSE_POINTS = list(range(27, 35))

# mouth points list (range)
MOUTH_POINTS = list(range(48, 61))

# jaw points list (range)
JAW_POINTS = list(range(0,17))

# points used for lining up the images
IMAGE_ALIGN_POINTS = (LEFT_EYEBROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_EYEBROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# overlay points other picture
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_EYEBROW_POINTS + RIGHT_EYEBROW_POINTS, NOSE_POINTS + MOUTH_POINTS,]

COLOR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

############ EXPECTETIONS ############

class NoFaces(Exception):
    pass

class ToManyFaces(Exception):
    pass
    

############ Finding Landmarks ############

def Finde_Landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise ToManyFaces
    if len(rects) == 0:
        raise NoFaces
        
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
 

############ Annotate Landmarks ############ 
 
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0,255,255))
    return im
    
def Draw_ConvexHull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color = color)

def getFaceMask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype = numpy.float64)
    
    for group in OVERLAY_POINTS:
        Draw_ConvexHull(im, landmarks[group], color = 1)
              
    im = numpy.array([im, im, im]).transpose((1, 2, 0))
    
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    
    return im
    
def transfromFromPoints(points1, points2):
    
    points1 = points1.astype(numpy.float64)
    points2 = points1.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    
    R = (U *Vt).T
    
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])
    

def readimageAndLandmarks(image):
    im = image
    im - cv2.resize(im,None, fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    
    s = Finde_Landmarks(im)
    
    return im, s
    
def warp_image(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype = im.dtype)
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst = output_im, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP)
    
    return output_im
    
def colorCorrect(im1, im2, landmarks1):
    blur_amount = COLOR_CORRECT_BLUR_FRAC * numpy.linalg.norm(numpy.mean(landmarks1[LEFT_EYE_POINTS], axis = 0) - numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis = 0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    
    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) / im2_blur.astype(numpy.float64))
    
def swappit(image1, image2):
    
    im1, landmarks1 = readimageAndLandmarks(image1)
    im2, landmarks2 = readimageAndLandmarks(image2)
    
    M = transfromFromPoints(landmarks1[IMAGE_ALIGN_POINTS], landmarks2[IMAGE_ALIGN_POINTS])
    
    mask = getFaceMask(im2, landmarks2)
    warped_mask  = warp_image(mask, M, im1.shape)
    combined_mask = numpy.max([getFaceMask(im1, landmarks1), warped_mask], axis = 0)
    
    warped_im2 = warp_image(im2, M, im1.shape)
    warp_corrected_im2 = colorCorrect(im1, warped_im2, landmarks1)
    
    output_im = im1 * (1.0 - combined_mask) + warp_corrected_im2 * combined_mask
    cv2.imwrite('output.jpg', output_im)
    image = cv2.imread('output.jpg')
    
    return image
    
    
image1 = cv2.imread(str(firstPicPath))
image2 = cv2.imread(str(secondPicPath))

swapped = swappit(image1, image2)
cv2.imshow('Face Swap 1:', swapped)

swapped = swappit(image2, image1)
cv2.imshow('Face Swap 2:', swapped)

cv2.waitKey(0)

cv2.destroyAllWindows()

