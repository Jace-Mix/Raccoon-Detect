from python.non_max_suppress import non_max_suppress
from python import config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2

# Argument Construction
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# Load model and encoder
print('Loading model and encoder...')
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.ENCODER_PATH, 'rb').read())

# Load image
image = cv2.imread(args['image'])
image = imutils.resize(image, width=500)

# Use selective search for region proposals
print('Running selective search...')
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

# Initialization
proposals = []
boxes = []

# Iterate through boxes
for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
    
    # Extract region and convert
    roi = image[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
    
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
    
    # Add to lists
    proposals.append(roi)
    boxes.append((x, y, x+w, y+h))

# Convert to numpy
proposals = np.array(proposals, dtype='float32')
boxes = np.array(boxes, dtype='int32')

# Classify each proposal
print('Classifying proposals...')
proba = model.predict(proposals)

# Find indices for positive results
print('Extracting positive results...')
labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == 'raccoon')[0]

# Retrieve bounding boxes and associated class label probabilities
boxes = boxes[idxs]
proba = proba[idxs][:, 1]

# Filter through boxes
idxs = np.where(proba >= config.MIN_PROB)
boxes = boxes[idxs]
proba = proba[idxs]

# Visualize WITHOUT NMS
clone = image.copy()

# Loop over bounding boxes
for (box, prob) in zip(boxes, proba):
    
    # Draw boxes
    (startX, startY, endX, endY) = box
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = 'Raccoon: {:.2f}%'.format(prob * 100)
    cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Display
cv2.imshow('Before NMS', clone)

# Run NMS algorithm
boxIdxs = non_max_suppress(boxes, proba)

# Loop over bounding boxes
for i in boxIdxs:
    
    # Draw boxes
    (startX, startY, endX, endY) = boxes[i]
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = 'Raccoon: {:.2f}%'.format(proba[i] * 100)
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Display with NMS
cv2.imshow('After NMS', image)
cv2.waitKey(0)