# Box A = Ground Truth, Box B is the predicted box
def compute_iou(boxA, boxB):
    
    # Calculate coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Calculate intersection
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Calculate area for Union
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute IoU
    retval = intersection / float(boxAArea + boxBArea - intersection)
    
    return retval