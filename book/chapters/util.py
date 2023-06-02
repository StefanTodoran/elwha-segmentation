import numpy as np

def getSavePath(type, number) -> str:
  if (type == "RGB"):
    return f"../data/img/airborne_{number + 1}.png"
  if (type == "IR"):
    return f"../data/img/airborne_ir_{number}.png"
  
  if (type == "STITCH"):
    return f"../out/intermediate/stitch_{number + 1}.png"
  if (type == "MATCH"):
    return f"../out/intermediate/match_{number + 1}.png"
  if (type == "KEYPOINTS"):
    return f"../out/intermediate/keypoints_{number + 1}.png"
  if (type == "MASK"):
    return f"../out/intermediate/mask_{number + 1}.png"

  if (type == "FINAL"):
    return f"../out/realignment/rgb_{number + 1}.png"
  else:
    raise ValueError(f"{type} is not a recognized save type!")

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

def onlyNumeric(seq):
  seq = str(seq)
  seq_type= type(seq)
  return seq_type().join(filter(seq_type.isdigit, seq))