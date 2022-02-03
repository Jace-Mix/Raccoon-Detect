from python.intersect_over_union import compute_iou
from python import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

# Create dataset
for dirPath in (config.POSITIVE_PATH, config.NEGATIVE_PATH):
    # Create output directory
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    
# Locate images
imagePaths = list(paths.list_images(config.ORIG_IMAGES))
    
# Initialize pos and neg counts
totalPositive = 0
totalNegative = 0

for (i, imagePath) in enumerate(imagePaths):
    
    # Progression
    print('processing image: {}/{}'.format(i + 1, len(imagePaths)))
    
    # Grab filename to use for path to XML files
    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind('.')]
    annotPath = os.path.sep.join([config.ORIG_ANNOTS, '{}.xml'.format(filename)])
    
    # Load XML files with ground truths
    contents = open(annotPath).read()
    soup = BeautifulSoup(contents, 'html.parser')
    gtBoxes = []
    
    # Extract width and height
    w = int(soup.find('width').string)
    h = int(soup.find('height').string)
    
    for o in soup.find_all('object'):
        
        # Get Bounding Box coordinates
        label = o.find('name').string
        xMin = int(o.find('xmin').string)
        yMin = int(o.find('ymin').string)
        xMax = int(o.find('xmax').string)
        yMax = int(o.find('ymax').string)
        
        # Shape up the box in case it is too big for the image
        xMin = max(0, xMin)
        yMin = max(0, yMin)
        xMax = min(w, xMax)
        yMax = min(h, yMax)
        
        # Put the box in our list
        gtBoxes.append((xMin, yMin, xMax, yMax))
    
    # Load image
    image = cv2.imread(imagePath)
    
    # Perform Selective Search with OpenCV
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposedRects = []
    
    # Go through all rectangles and populate the proposed rectangles list
    for (x, y, w, h) in rects:
        # Convert boxes from start to end
        proposedRects.append((x, y, x+w, y+h))
    
    # Initialize ROIs
    positiveROIs = 0
    negativeROIs = 0
    
    # Go through max number of proposals
    for proposedRect in proposedRects[:config.MAX_PROPOSALS]:
        
        # Retrieve rectangle box
        (propStartX, propStartY, propEndX, propEndY) = proposedRect
        
        # Go through ground-truth boxes
        for gtBox in gtBoxes:
            # Compute IOU
            iou = compute_iou(gtBox, proposedRect)
            (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox
            
            roi = None
            outputPath = None
            
            # If IOU is better than 70% and we haven't reached our pos limit
            if iou > 0.7 and positiveROIs <= config.MAX_POSITIVE:
                
                # Extract ROI and save
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = '{}.png'.format(totalPositive)
                outputPath = os.path.sep.join([config.POSITIVE_PATH, filename])
                
                # Increment
                positiveROIs += 1
                totalPositive += 1
            
            # What if the proposed box was within the ground truth box?
            fullOverlap = propStartX >= gtStartX
            fullOverlap = fullOverlap and propStartY >= gtStartY
            fullOverlap = fullOverlap and propEndX <= gtEndX
            fullOverlap = fullOverlap and propEndY <= gtEndY
            
            # No full overlap and IOU is less than 5% and we haven't reached our capacity
            if not fullOverlap and iou < 0.05 and negativeROIs <= config.MAX_NEGATIVE:
                
                # Extract ROI and save
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = '{}.png'.format(totalNegative)
                outputPath = os.path.sep.join([config.NEGATIVE_PATH, filename])
                
                # Increment
                negativeROIs += 1
                totalNegative += 1
                
            # Are the paths valid?
            if roi is not None and outputPath is not None:
                
                # Resize ROI and export
                roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)