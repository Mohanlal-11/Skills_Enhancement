import torch
from pathlib import Path
import numpy as np

def calculate_iou(bbox1, bbox2, format="corners"):
  """This is a function that calculates the intersection over union between two bounding boxes
  Parameters/Arguments:
    bbox1: coordinates of first bounding box
    bbox2: coordinates of second bounding box
    format: format of bounding box i.e.

      if format == "corners":
        bbox = [..., x_min, y_min, x_max, y_max]
      if format == "center":
        bbox = [..., x_center, y_center, width, height]

  Returns the intersection over union score i.e. between 0 to 1 which has the shape of [batch_size(number of bounding boxes), iou_score]
    """

  #Extract the co-ordinates of bounding boxes in [x_min, y_min, x_max, y_max] format irrespective of given format.
  if format == "corners":
    x1 = bbox1[..., 0:1]
    y1 = bbox1[..., 1:2] #(x1, y1) --> upper left corner of bbox1
    x2 = bbox1[..., 2:3]
    y2 = bbox1[..., 3:4] #(x2, y2) --> bottom right corner of bbox1

    X1 = bbox2[..., 0:1]
    Y1 = bbox2[..., 1:2] #(X1, Y1) --> upper left corner of bbox1
    X2 = bbox2[..., 2:3]
    Y2 = bbox2[..., 3:4] #(X2, Y2) --> bottom right corner of bbox1

  elif format == "center":
    x1 = bbox1[..., 0:1] - (bbox1[..., 2:3] / 2)
    y1 = bbox1[..., 1:2] - (bbox1[..., 3:4] / 2) #(x1, y1) --> upper left corner of bbox1
    x2 = bbox1[..., 0:1] + (bbox1[..., 2:3] / 2)
    y2 = bbox1[..., 1:2] + (bbox1[..., 3:4] / 2) #(x2, y2) --> bottom right corner of bbox1

    X1 = bbox2[..., 0:1] - (bbox2[..., 2:3] / 2)
    Y1 = bbox2[..., 1:2] - (bbox2[..., 3:4] / 2) #(X1, Y1) --> upper left corner of bbox1
    X2 = bbox2[..., 0:1] + (bbox2[..., 2:3] / 2)
    Y2 = bbox2[..., 1:2] + (bbox2[..., 3:4] / 2) #(X2, Y2) --> bottom right corner of bbox1

  #Calculate the area of intesection between two bounding boxes.
  a = torch.max(x1, X1)
  b = torch.max(y1, Y1) # co-ordinates of upper left corner of intersected region
  c = torch.min(x2, X2)
  d = torch.min(y2, Y2) # co-ordinates of bottom right corner of intersected region

  W = (c-a).clamp(0)
  H = (d-b).clamp(0)
  intersection_area = W * H  # Area of intersection of two bounding boxes

  #Calculate the area of union between two bounding boxes.
  bbox1_area = abs((x2-x1) * (y2-y1)) # (x2-x1) gives width of bbox and (y2-y1) gives height of bbox
  bbox2_area = abs((X2-X1) * (Y2-Y1)) # These i.e. bbox1_area and bbox2_area both are the total area of two given bboxes.

  union_area = bbox1_area + bbox2_area - (intersection_area)

  IoU = (intersection_area / union_area)

  return IoU

# if __name__ =='__main__':
#   a1 = torch.tensor([[4.0, 6.0, 4.0, 4.0],
#                    [4.0, 5.0, 4.0, 4.0]]) # x, y, w, h
#   a2 = torch.tensor([[6.0, 4.0, 6.0, 2.0],
#                    [6.0, 3.0, 6.0, 2.0]])
#   iou = calculate_iou(a1, a2, format="center")
#   print(f"The iou  is : {iou} and shape of outputed iou is: {iou.shape}")


def nms(pred_bboxes:list,
        prob_threshold:float,
        iou_threshold:float,
        format:str):
  """ This is the function that performs the non-maximum suppression between predicted bounding boxes form the model, which means it is used in post-processing.

  Parameters/Argumenst:
  pred_bboxes: It is the list that contains predicted bounding boxes.
                i.e. [[confidence_score, x_min, y_min, x_max, y_max, class_probabilities],[],................,845] ; here, the co-ordinates of bounding box could be in different format.
  prob_threshold: Threshold value to select the few predicted bounding boxes which may contains object.
  iou_threshold: Threshold value to select only one bb of a class.
  format: It is the format of bounding box representation i.e. 'corners' or 'center'.

  Returns the list of bounding boxes for each class which contains the object.
  """
  assert type(pred_bboxes)==list, f"The given pred bboxes are not list, instead they are {type(pred_bboxes)}"
  bboxes = [box for box in pred_bboxes if box[4] > prob_threshold] #This is list comprehension that keeps only the bboxes which has confidence/probability score higher than given threshold.
  bboxes = sorted(bboxes, key=lambda x:x[4], reverse=True) # This keeps the bounding boxes in descending order according to its confidence/probability score.
  selected_bboxes = [] # To store the bboxes that have highest confidence score for each class

  while bboxes:
    choosen_box = bboxes.pop(0) # pop out the bbox of the first index of that sorted list to choosen_box variable and with this we are going to calculate iou with others.

    #Now in that sorted list, check each bbox has same class with choosen one or not and check iou between that box and choosen is less than given iou_threshold or not,
    #If the bbox and choosen bbox are in same class and have higher iou then they are not included in list.
    #If they are in different class and have lower iou then they are included in that list and that will be again checked in next iteration.

    bboxes = [box for box in bboxes if torch.argmax(torch.tensor(box[5:])) != torch.argmax(torch.tensor(choosen_box[5:])) or calculate_iou(torch.tensor(box[0:4]), torch.tensor(choosen_box[0:4]), format="center")<iou_threshold]
    selected_bboxes.append(choosen_box) #This list contains the bbox that have high probability of object of a certain class.
  return selected_bboxes


def mAP(pred_bboxes:list,
        ground_truth_bboxes:list,
        iou_threshold=0.5,
        format="center",
        num_classes:int=20):
  """This function calculates the mean average precision of the model.

  Parameters/Arguments:
    Pred_bboxes: These are the bounding boxes predicted by our object detection model and here its format is list i.e. [[5+number of classes], [], [], .............,845], here 845 means for the input of 416x416 and 5 anchor boxes we have (13*13*5)=845 boxes for each image..
    ground_truth_bboxes: These are the ground truth bounding boxes to calculate mAP and here its format is also same as pred_bboxes.
    iou_threshold: Threshold value that is used to find the true positive and false positive.
    num_class: Number of class that our model can detect and classify.

  Returns the mean average precision value of the model.
  """
  average_precision = [] #To store the average precisions of each class so that we can calculate mAP at last.
  assert type(pred_bboxes) == list, f"The given predicted bounding boxes are not in list format instead {type(pred_bboxes)}"
  assert type(ground_truth_bboxes) == list, f"The given ground_truth bounding boxes are not in list format instead {type(ground_truth_bboxes)}"
  for i in range(num_classes):
    predicted = [] #To store the predicted bounding boxes of each class in each iteration .
    label = [] #To store the ground truth label bounding boxes of each class in each iteration.

    #Start a for loop to check the class of each predicted bounding boxes and store it in a list
    for p_bbox in pred_bboxes:
      if p_bbox[5:].index(max(p_bbox[5:])) == i:
        predicted.append(p_bbox)
    #Same for ground truth label/bounding boxes
    for g_bbox in ground_truth_bboxes:
      if g_bbox[5:].index(max(g_bbox[5:])) == i:
        label.append(g_bbox)

    len_true_bbox = len(label)
    len_pred_bbox = len(predicted)
    if (len_true_bbox * len_pred_bbox) ==0:
      continue

    predicted = sorted(predicted, key=lambda x:x[0], reverse=True) #Sorting i.e. in descending order the predicted boxes of a each class according to its probability score
    label = sorted(label, key=lambda x:x[0], reverse=True)
    TP = []
    FP = []

    for i, pred in enumerate(predicted):#Now calculate iou between each predicted and all ground_truth_label bboxes of a class
      ious = []
      for truth_box in label:
        iou = calculate_iou(torch.tensor(pred[1:5]),
                            torch.tensor(truth_box[1:5]),
                            format=format)
        max_iou = torch.max(iou) # Selects a best iou among the iou between one predicted and all ground truth label.
        ious.append(max_iou.item())
      #Append list of true positive and false positive with 1 and 0 respectively if best iou is greater than given iou_threshold otherwise reverse.
      best_iou = max(ious)
      if best_iou > iou_threshold:
        TP.append(1)
        FP.append(0)
      else:
        FP.append(1)
        TP.append(0)
    #Calculate cumulative sum of true positive and false positive for each classes
    TP_cumulative_sum = torch.cumsum(torch.tensor(TP), dim=0)
    FP_cumulative_sum = torch.cumsum(torch.tensor(FP), dim=0)
    #Calculate precision and recall for each classes
    precision = TP_cumulative_sum / (TP_cumulative_sum + FP_cumulative_sum)
    recall = TP_cumulative_sum / (len_true_bbox)

    #The Precision-Recall (P-R) curve typically spans from recall 0 to recall 1. By appending a precision of 1 at recall 0, we ensure that the curve begins at a well-defined starting point, which is important for the area calculation
    precision = torch.cat((torch.tensor([1]), precision))
    recall = torch.cat((torch.tensor([0]), recall))

    #Now the area under the precision-recall curve is calculated using the torch.trapz i.e. "trapezoidal rule (integration)"
    AUC = torch.trapz(precision, recall)
    average_precision.append(AUC)
  if len(average_precision) != 0:
    map = sum(average_precision) / len(average_precision) #Since mAP is the average of average precision(area under the curve)
  elif len(average_precision) == 0:
    map = 0
  return map

def set_grid(grid_size:int):
  grid_x, grid_y = torch.meshgrid((torch.arange(grid_size), torch.arange(grid_size)), indexing='ij')
  return grid_x, grid_y

def set_grid_xy(grid_size:int):
  #Set/generate the co-ordinates for each grid cell of final output featuer map's size
  grid_x, grid_y =  set_grid(grid_size=grid_size)
  grid_x = grid_x.contiguous().view(1, grid_size, grid_size, 1).expand(1,grid_size, grid_size,3) #Here, '.contiguous()' : This creates a new tensor with the same data but in a contiguous memory layout. The new tensor is a copy and not a view.
  grid_y = grid_y.contiguous().view(1, grid_size, grid_size, 1).expand(1,grid_size, grid_size,3) #Shape of both grid is : [1,13,13,1]
  return grid_y, grid_x

def transform_predicted_txtytwth(bboxes:torch.Tensor, grid_size:list, device:torch.device, scale:str):
  """
  This function converts the co-ordinates of bounding boxes relative to grid cell into the co-ordinates of bboxes relative to anchor boxes.

  Parameters/Arguments:
    bboxes: Co-ordinates of bounding boxes i.e. [[..., to, tx, ty, w, h, class_prob]].
    anchors: Pre-defined boxes to predict bounding boxes.
    grid_size: size of final grid/feature map.
    device: It is the device on which predictions are located.

  Returns the co-ordinates of bounding boxes in the format (x_center, y_center, width, height).
  """
  #First extract the co-ordinates of bboxes.
  
  tx = bboxes[..., 0]
  ty = bboxes[..., 1]
  w = bboxes[..., 2]
  h = bboxes[..., 3] #shape: [batch_size, gird_x, grid_y, no. of anchor boxes]
  to = bboxes[..., 4]
  c = bboxes[..., 5:]
  # print(f"shape of outputs : {tx.shape, ty.shape, w.shape, h.shape}")
  anchors = [
            [116,90, 156,198, 373,326],
            [30,61, 62,45, 59,119],
            [10,13, 16,30, 33,23]]
  large_scale_anchor = torch.tensor(anchors[0][:])
  large_scale_anchor_w = large_scale_anchor[::2].view(1,1,1,3).expand(1,19,19,3).to(device)
  large_scale_anchor_h = large_scale_anchor[1::2].view(1,1,1,3).expand(1,19,19,3).to(device)
  # print(f"shape of large anchor w: {large_scale_anchor_w.shape}, {large_scale_anchor_w}")
  
  medium_scale_anchor = torch.tensor(anchors[1][:])
  medium_scale_anchor_w = medium_scale_anchor[::2].view(1,1,1,3).expand(1,38,38,3).to(device)
  medium_scale_anchor_h = medium_scale_anchor[1::2].view(1,1,1,3).expand(1,38,38,3).to(device)
  small_scale_anchor = torch.tensor(anchors[2][:])
  small_scale_anchor_w = small_scale_anchor[::2].view(1,1,1,3).expand(1,76,76,3).to(device)
  small_scale_anchor_h = small_scale_anchor[1::2].view(1,1,1,3).expand(1,76,76,3).to(device)

  #Now convert the predicted co-ordinates of bboxes into format (x_center, y_center, width, height) according paper: https://arxiv.org/abs/1612.08242 Figure3
  if scale == "large":
    grid_x_large, grid_y_large = set_grid_xy(grid_size=grid_size[0])
    # print(f"grid_x: {grid_x_large}, {grid_x_large.shape}")
    # print(f"grid_y: {grid_y_large}, {grid_y_large.shape}")
    to = torch.sigmoid(to)
    x_c = torch.sigmoid(tx) + grid_x_large.to(device)
    y_c = torch.sigmoid(ty) + grid_y_large.to(device)
    w = torch.exp(w) *large_scale_anchor_w
    h = torch.exp(h) * large_scale_anchor_h

    coordinates = torch.stack((x_c, y_c, w, h, to), dim=-1) #here, stack will stack in the co-ordinates into column/verrically (adds in last dimension) and its shape is: [169, 5]+[169, 5]+[169, 5]+[169, 5] = [169, 5, 4] ---> this means we have 5 bounding boxes with 4 co-ordinates in each 169 pixels/grid_cell of predicted/final output layer.
    return torch.cat((coordinates, c), dim=-1)
  
  if scale == "medium":
    grid_x_medium, grid_y_medium  = set_grid_xy(grid_size=grid_size[1])
    to = torch.sigmoid(to)
    x_c = torch.sigmoid(tx) + grid_x_medium.to(device)
    y_c = torch.sigmoid(ty) + grid_y_medium.to(device)
    w = torch.exp(w) *medium_scale_anchor_w
    h = torch.exp(h) * medium_scale_anchor_h
    coordinates = torch.stack((x_c, y_c, w, h, to), dim=-1) #here, stack will stack in the co-ordinates into column/verrically (adds in last dimension) and its shape is: [169, 5]+[169, 5]+[169, 5]+[169, 5] = [169, 5, 4] ---> this means we have 5 bounding boxes with 4 co-ordinates in each 169 pixels/grid_cell of predicted/final output layer.
    return torch.cat((coordinates, c), dim=-1)

  if scale == "small":
    grid_x_small, grid_y_small = set_grid_xy(grid_size=grid_size[2])
    to = torch.sigmoid(to)
    x_c = torch.sigmoid(tx) + grid_x_small.to(device)
    y_c = torch.sigmoid(ty) + grid_y_small.to(device)
    w = torch.exp(w) *small_scale_anchor_w
    h = torch.exp(h) * small_scale_anchor_h

    coordinates = torch.stack((x_c, y_c, w, h, to), dim=-1) #here, stack will stack in the co-ordinates into column/verrically (adds in last dimension) and its shape is: [169, 5]+[169, 5]+[169, 5]+[169, 5] = [169, 5, 4] ---> this means we have 5 bounding boxes with 4 co-ordinates in each 169 pixels/grid_cell of predicted/final output layer.
    return torch.cat((coordinates, c), dim=-1)

def save_weights(model:torch.nn.Module,
                 save_dir:str,
                 model_name:str):
  """
  This function is to save the weights of trained model yolov2.

  Parameters:
    model: Trained yolov2 model.
    save_dir: Name of directory in which model's weight is to be save.
    model_name: Name of model's weight in ".pt" or ".pth" format.
  """
  dir = Path(save_dir)
  dir.mkdir(parents=True, exist_ok=True)

  save_path = dir / model_name
  assert model_name.endswith(".pt") or model_name.endswith(".pth"), f"The given model name's format is not valid. Please save model with valid name such as 'xyz.pt' or 'xyz.pth'"

  torch.save(obj=model.state_dict(), f=save_path)
  print(f"[INFO] Your model's weights are saved to directory {dir} with name {model_name}")
