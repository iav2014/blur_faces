'''
anonymized face people in realtime video (web cam)
using pyimagesearch functions applied to webcam frame
(c) Nacho Ariza 2020
'''
import cv2
import os
import imutils
import numpy as np
from anonymized.face_blurring import anonymize_face_pixelate
from anonymized.face_blurring import anonymize_face_simple

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")

prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
g_confidence = 0.5
method = "pixelated" # simple or pixelated
blocks = 6 # if pixelated, say blocks number, more block number = mode definition = bad blur result

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
  rval, frame = vc.read()
else:
  rval = False
# main loop
while rval: # infinite loop until rval = false
  cv2.imshow("anonymizer", frame)
  # read image from camera
  rval, frame = vc.read()
  key = cv2.waitKey(20)
  if key == 27:  # exit on ESC key
    break
  frame = imutils.resize(frame, width=400)
  
  # grab the dimensions of the frame and then construct a blob
  # from it
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                               (104.0, 177.0, 123.0))
  
  # pass the blob through the network and obtain the face detections
  net.setInput(blob)
  detections = net.forward()
  
  # loop over the detections
  for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the detection
    confidence = detections[0, 0, i, 2]
    
    # filter out weak detections by ensuring the confidence is
    # greater than the minimum confidence
    if confidence > g_confidence:
      # compute the (x, y)-coordinates of the bounding box for
      # the object
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      
      # extract the face ROI
      face = frame[startY:endY, startX:endX]
      
      # check to see if we are applying the "simple" face
      # blurring method
      if method == "simple":
        face = anonymize_face_simple(face, factor=3.0)
      
      # otherwise, we must be applying the "pixelated" face
      # anonymization method
      else:
        face = anonymize_face_pixelate(face,
                                       blocks=blocks)
      
      # store the blurred face in the output image
      frame[startY:endY, startX:endX] = face
cv2.destroyWindow("preview")
