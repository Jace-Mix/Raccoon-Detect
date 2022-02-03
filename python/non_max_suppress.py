import numpy as np

def non_max_suppress(boxes, prob):
    
    # Edge case: no boxes available
    if len(boxes) == 0:
        return []
    
    box_list = []
    threshold = 0.2
    
    # Coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # area calculations
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = prob.argsort()[::-1]
    
    while order.size > 0:
        
        i = order[0]
        box_list.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (area[i] + area[order[1:]] - inter)
        
        inds = np.where(ovr <= threshold)[0]
        
        order = order[inds + 1]
        
    return box_list