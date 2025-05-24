import torch
from torch import nn
from utils import calculate_iou

class Yolov3Loss(nn.Module):
  """
  This constructs the loss function to calculate the loss of bounding boxes in yolov3
  """
  def __init__(self, anchors:list, input_size:int):
    super(Yolov3Loss, self).__init__()
    self.input_size = input_size
    self.anchors = anchors

    self.noobj_loss_func = nn.BCEWithLogitsLoss()
    self.obj_box_loss_func = nn.MSELoss()
    self.class_loss_func = nn.CrossEntropyLoss()
    self.sigmoid = nn.Sigmoid()

    #These scale/lambda values are taken from others repo who have already implemented it from scratch using pytorch
    self.noobj_scale = 1
    self.obj_scale = 10
    self.box_scale = 14
    self.class_scale = 1
  def forward(self, predictions:torch.Tensor, labels:torch.Tensor, scale:str, device:torch.device):
    object_present = (labels[..., 0:1] == 1).float() #Here, if labels contains object then it will be equal to '1' and object_present will be all 1 of size [1,8,8,3,5+class] for large scale otherwise all 0.
    object_absence = (labels[..., 0:1] == 0).float() #Here, if labels doesn't contain object then it will be equal to '0' and object_absence will be all 1 of size [1,8,8,3,5+class] for large scale otherwise all 0.
    large_scale_anchor = torch.tensor(self.anchors[0][:])
    large_scale_anchor = large_scale_anchor.reshape((1,1,1,3,2)).to(device)
    medium_scale_anchor = torch.tensor(self.anchors[1][:])
    medium_scale_anchor = medium_scale_anchor.reshape((1,1,1,3,2)).to(device)
    small_scale_anchor = torch.tensor(self.anchors[2][:])
    small_scale_anchor = small_scale_anchor.reshape((1,1,1,3,2)).to(device) #Since, the yolov3 model predicts in 3 stages or 3 scaled output for 3 bounding boxes.
    if scale == "large":
      no_obj_loss = self.noobj_loss_func(predictions[..., 0:1] * object_absence, labels[..., 0:1] * object_absence)

      pred_box_center = self.sigmoid(predictions[..., 1:3])
      pred_box_wh = large_scale_anchor * torch.exp(predictions[..., 3:5])# Here, when model predicts the output its has bounding boxes whose center are normalized [0,1] and width, height are relative to grid size and in its line number 49 we can see it is again divided by anchor and took log so here it is raised to exp and multiplied by anchor an for center the ground truth is normalize i.e. between [0,1] so took sigmoid.
      predicted_box = torch.cat((pred_box_center, pred_box_wh), dim=-1)
      iou_pred_label = calculate_iou(predicted_box, labels[..., 1:5], format="center").detach()

      #While calculating object loss, the confidence score of prediction should be pass through sigmoid and target's confidence score should be multiplied with iou
      object_loss = self.obj_box_loss_func(predictions[..., 0:1] * object_present, (labels[..., 0:1] * object_present) * iou_pred_label)

      pred_xy = self.sigmoid(predictions[..., 1:3])
      pred_wh = predictions[..., 3:5] # Here, the center coordinates should be between 0 and 1 because its ground truth is normalized and at certain(x) % of grid cell size but width height are relative to grid size and divide by anchors.
      pred_box = torch.cat((pred_xy, pred_wh), dim=-1)

      label_xy = labels[..., 1:3]
      label_wh = torch.log(labels[..., 3:5] / large_scale_anchor + 1e-16) # This is according to paper and also ground truth needs to be mapped into the same scale as the predicted
      label_box = torch.cat((label_xy, label_wh), dim=-1)

      box_loss = self.obj_box_loss_func(pred_box * object_present, label_box * object_present)

      class_loss = self.class_loss_func(predictions[..., 5:] * object_present, labels[..., 5:] * object_present) #since classes are one hot encoded i.e. only one class is high in a bounding box and other are low.

      combined_large_scale_loss = (
          self.noobj_scale * no_obj_loss
          + self.obj_scale * object_loss
          + self.box_scale * box_loss
          + self.class_scale * class_loss
      )
      # print(no_obj_loss, object_loss, box_loss, class_loss)
      # print(combined_large_scale_loss)
      return combined_large_scale_loss

    if scale == "medium":
      no_obj_loss = self.noobj_loss_func(predictions[..., 0:1] * object_absence, labels[..., 0:1] * object_absence)

      pred_box_center = self.sigmoid(predictions[..., 1:3])
      pred_box_wh = medium_scale_anchor * torch.exp(predictions[..., 3:5])
      predicted_box = torch.cat((pred_box_center, pred_box_wh), dim=-1)
      iou_pred_label = calculate_iou(predicted_box, labels[..., 1:5], format="center").detach()

      object_loss = self.obj_box_loss_func(predictions[..., 0:1] * object_present, (labels[..., 0:1] * object_present) * iou_pred_label)

      pred_xy = self.sigmoid(predictions[..., 1:3])
      pred_wh = predictions[..., 3:5]
      pred_box = torch.cat((pred_xy, pred_wh), dim=-1)

      label_xy = labels[..., 1:3]
      label_wh = torch.log(labels[..., 3:5] / medium_scale_anchor + 1e-16)
      label_box = torch.cat((label_xy, label_wh), dim=-1)

      box_loss = self.obj_box_loss_func(pred_box * object_present, label_box * object_present)

      class_loss = self.class_loss_func(predictions[..., 5:] * object_present, labels[..., 5:] * object_present)

      combined_medium_scale_loss = (
          self.noobj_scale * no_obj_loss
          + self.obj_scale * object_loss
          + self.box_scale * box_loss
          + self.class_scale * class_loss
      )
      # print(no_obj_loss, object_loss, box_loss, class_loss)
      return combined_medium_scale_loss

    if scale == "small":
      no_obj_loss = self.noobj_loss_func(predictions[..., 0:1] * object_absence, labels[..., 0:1] * object_absence)

      pred_box_center = self.sigmoid(predictions[..., 1:3])
      pred_box_wh = small_scale_anchor * torch.exp(predictions[..., 3:5])
      predicted_box = torch.cat((pred_box_center, pred_box_wh), dim=-1)
      iou_pred_label = calculate_iou(predicted_box, labels[..., 1:5], format="center").detach()

      object_loss = self.obj_box_loss_func(predictions[..., 0:1] * object_present, (labels[..., 0:1] * object_present) * iou_pred_label)

      pred_xy = self.sigmoid(predictions[..., 1:3])
      pred_wh = predictions[..., 3:5]
      pred_box = torch.cat((pred_xy, pred_wh), dim=-1)

      label_xy = labels[..., 1:3]
      label_wh = torch.log(labels[..., 3:5] / small_scale_anchor + 1e-16)
      label_box = torch.cat((label_xy, label_wh), dim=-1)

      box_loss = self.obj_box_loss_func(pred_box * object_present, label_box * object_present)

      class_loss = self.class_loss_func(predictions[..., 5:] * object_present, labels[..., 5:] * object_present)

      combined_small_scale_loss = (
          self.noobj_scale * no_obj_loss
          + self.obj_scale * object_loss
          + self.box_scale * box_loss
          + self.class_scale * class_loss
      )
      # print(no_obj_loss, object_loss, box_loss, class_loss)
      return combined_small_scale_loss

    else:
      print(f"[INFO] The given scale [{scale}] is wrong argument, enter right scale i.e. |'large'|, |'medium'|, |'small'| ")

if __name__ == "__main__":
  from yolov3_model import YOLOv3
  anchors_list = [
            [116,90, 156,198, 373,326],
            [30,61, 62,45, 59,119],
            [10,13, 16,30, 33,23]]
  num_class = 80
  input_size = 256
  label_large = torch.randn((1,8,8,3,5+num_class))
  label_medium = torch.randn((1,16,16,3,5+num_class))
  label_small = torch.randn((1,32,32,3,5+num_class))
  input_img = torch.randn((1,3,input_size,input_size))

  model = YOLOv3(input_size=input_size, anchors=anchors_list, num_classes=num_class)
  pred_large, pred_medium, pred_small = model(input_img)
  loss_fn = Yolov3Loss(anchors=anchors_list, input_size=input_size)
  loss = (
      loss_fn(predictions=pred_large, labels=label_large, scale="large")
      + loss_fn(predictions=pred_medium, labels=label_medium, scale="medium")
      + loss_fn(predictions=pred_small, labels=label_small, scale="small")
  )
  print(f"The loss for the random/dummy input is: {loss}")
