import numpy as np

def percentage(loss, total):
  if (hasattr(loss, '__len__')):
    return round(100 * len(loss) / total, 2)
  else:
    return round(100 * loss / total, 2)
  
def distance(n, m):
  return abs(np.sum(n - m))

# Used for cv2.decomposeHomographyMat, which returns the 
# arrays inside single element tuples for some reason?
def unTuple(d):
  _, r, t, n = d
  return r[0], t[0], n[0]

def isNearCenter(point, dimensions, threshold):
  dx_center = abs(point[0] - (dimensions[1] / 2)) / dimensions[1]
  dy_center = abs(point[1] - (dimensions[0] / 2)) / dimensions[0]

  return (dx_center < threshold) and (dy_center < threshold)

def getQuadrant(point, dimensions):
  left = True if point[0] < (dimensions[1] / 2) else False
  top = True if point[1] < (dimensions[0] / 2) else False

  if (top and left): return 0
  if (top and not left): return 1
  if (not top and not left): return 2
  if (not top and left): return 3

def withinThresholdDeviations(value, threshold, baseline):
  return value < baseline + threshold and value > baseline - threshold