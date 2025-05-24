import torch
from yolov3_model import YOLOv3
from utils import calculate_iou, nms, transform_predicted_txtytwth
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision import transforms
import os
import math
import cv2 as cv
from pathlib import Path
import argparse
import time

parser = argparse.ArgumentParser(description="This is for the arguments required for the inference of yolov3.")
parser.add_argument("-t", "--path", type=str, default="./imgs", help="Argument for the path to the directory of testing images.")
parser.add_argument("-s", "--save", type=str, default="./detected", help="Argument for the path to directory where we want to save out detected results.")
parser.add_argument("-n", "--nms_conf", type=float, default=0.5, help="Argument to provide confidence threshold to perform nms")
parser.add_argument("-p", "--plot_conf", type=float, default=0.70, help="Argument to plot the predictions which have given confidence.")
parser.add_argument("-i", "--input_size", type=int, default=608, help="Argument for the size of input image to model.")
parser.add_argument("-w", "--wts", type=str, default="/home/mohan/Desktop/Office/Learning session/yolov3_reference_wts/yolov3_wts_reference.pt", help="Argument for the path to trained weights.")

arg = parser.parse_args()

def visualize_prediction(
    model:torch.nn.Module,
    image_path:str,
    transform:transforms.Compose,
    iou_threshold:float,
    anchors:list,
    size:list):
  """
  This function is to visualize and save the predictions from the trained yolov3.
  Arguments:
    model(torch): yolov3 architecture in pytorch framework.
    image_path(str): path to the image on which the model makes predictions.
    transform(torchvision): transformation module to transform the input image into suitable format that model takes.
    iou_threshold(float): for nms.
    anchors(list): to perform post processing on predictions from model.
    grid(list): gird sizes of each scale's feature map.
  
  Plots and saves the bounding box plotted images in a particular directory.
  """
  # CLASSES = ["cat", "chair", "dog"]
  # CLASSES = [
  #   "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  #   "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
  #   "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
  #   "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
  #   "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  #   "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
  #   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
  #   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
  #   "mirror", "dining table", "window", "desk", "toilet", "door", "TV", "laptop", "mouse",
  #   "remote", "keyboard", "cell phone", "microwave", "oven", "toaster"]
  CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

  # image = Image.open(arg.path+"/"+image_path)
  # print(f"scales: {size}")
  image_ori = cv.imread(arg.path+"/"+image_path, cv.IMREAD_COLOR)
  image = cv.resize(image_ori, (608, 608))
  transformed_image = transform(image)
  input_image = transformed_image.unsqueeze(dim=0)
  batch = input_image.shape[0]
  num_anchors = 3
  num_classes = 80
  start_time = time.time()
  model.eval()
  with torch.inference_mode():
    out = model(input_image)
  device = out[0].device
  l = out[0]
  m = out[1]
  s = out[2]
  # l = l.contiguous().view(batch, 19, 19, num_anchors, 5+num_classes)
  # m = m.contiguous().view(batch, 38, 38, num_anchors, 5+num_classes)
  # s =  s.contiguous().view(batch, 76, 76, num_anchors, 5+num_classes)
  pred_bboxes_large = transform_predicted_txtytwth(bboxes=l, grid_size=size, device=device, scale="large")
  pred_bboxes_large = pred_bboxes_large.flatten(0,3)
  pred_bboxes_large = pred_bboxes_large.tolist()
  filtered_bboxes_large = nms(pred_bboxes=pred_bboxes_large, prob_threshold=arg.nms_conf, iou_threshold=iou_threshold, format="center")

  pred_bboxes_medium = transform_predicted_txtytwth(bboxes=m, grid_size=size, device=device, scale="medium")
  pred_bboxes_medium = pred_bboxes_medium.flatten(0,3)
  pred_bboxes_medium = pred_bboxes_medium.tolist()
  filtered_bboxes_medium = nms(pred_bboxes=pred_bboxes_medium, prob_threshold=arg.nms_conf, iou_threshold=iou_threshold, format="center")

  pred_bboxes_small = transform_predicted_txtytwth(bboxes=s, grid_size=size, device=device, scale="small")
  pred_bboxes_small = pred_bboxes_small.flatten(0,3)
  pred_bboxes_small = pred_bboxes_small.tolist()
  filtered_bboxes_small = nms(pred_bboxes=pred_bboxes_small, prob_threshold=arg.nms_conf, iou_threshold=iou_threshold, format="center")

  all_predictions = [filtered_bboxes_large, filtered_bboxes_medium, filtered_bboxes_small]
  # for box in pred_bboxes_large:
  #   print(box)
  # figure, axis = plt.subplots(1, figsize=(12,8))
  # # axis.imshow(transformed_image.permute(1,2,0))
  # axis.imshow(image)
  # print(f"in test: {l.shape}, {size[0]}, {transformed_image.shape}")
  for idx, scale_box in enumerate(all_predictions):
    for box in scale_box:
      # print(box)
      confidence = box[4]
      x_c = (box[0]/size[idx])
      y_c = (box[1]/size[idx])
      w = (box[2]/608)
      h = (box[3]/608)
      class_prob = box[5:]
      label = torch.argmax(torch.softmax(torch.tensor(class_prob), dim=-1),dim=-1)

      x_min = (x_c - (w/2))*image_ori.shape[1]
      y_min = (y_c - (h/2))*image_ori.shape[0]
      x_max = (x_c + (w/2))*image_ori.shape[1]
      y_max = (y_c + (h/2))*image_ori.shape[0]
      x_min = max(0, x_min)
      y_min = max(0, y_min)
      x_max = max(0, x_max)
      y_max = max(0, y_max)
      if confidence>arg.plot_conf:
        # print([x_c, y_c, w, h, confidence])
        # print(image_ori.shape)
        image_ori = cv.rectangle(image_ori, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0,0), 3)
        image_ori = cv.putText(image_ori, CLASSES[label], (int((x_min+x_max)/2), int(y_min)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv.LINE_AA)
    cv.imwrite(f"./detected/{image_path}", image_ori)
    # print(f"The prediction on given image {image_path} is saved to detected/{image_path}")
  #       print([confidence, x_c, y_c, w, h])
  #       # print(f"The given image is: {os.path.basename(image_path)} and the predictions on this image is: {CLASSES[label]}")
  #       rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='r', facecolor='none')
  #       axis.add_patch(rect)

  #       axis.text(x_min + 10, y_min, CLASSES[label], color='red', fontsize=10, backgroundcolor="white")
  # plt.axis(False)
  # dir = Path(arg.save)
  # dir.mkdir(parents=True, exist_ok=True)
  # save_path = os.path.join(dir, os.path.basename(image_path))
  # plt.savefig(save_path)
  # plt.close()
  end_time = time.time()
  total_taken_time = end_time-start_time
  # print("====================================================================================================")
  print(f"The total time taken by the model to make a prediction on image {os.path.basename(image_path)} is : {total_taken_time}Sec.")
  # print("====================================================================================================")
  # print(f"[INFO] The prediction from yolov3 trained from scratch is plotted successfully........!")

def main(image_path:list, weight_path=arg.wts):
  """This function prepares the image paths and yolov3 model."""
  anchors_list = [
            [116,90, 156,198, 373,326],
            [30,61, 62,45, 59,119],
            [10,13, 16,30, 33,23]]
  iou_threshold = 0.3
  grid_size = [arg.input_size//32, arg.input_size//16, arg.input_size//8]
  num_class = 80

  transformation = transforms.Compose([
      transforms.ToTensor()
  ])

  wts = torch.load(f=weight_path, map_location=torch.device('cpu'), weights_only=True)
  # print(len(wts.keys()))
  # print("###################################################################################3")
  model = YOLOv3(input_size=arg.input_size, anchors=anchors_list, num_classes=num_class)
  # print(len(model.state_dict().keys()))
  new_wts = {my_para_name:wts[pretrained_para_name] for my_para_name, pretrained_para_name in zip(model.state_dict().keys(), wts.keys())}
  model.load_state_dict(state_dict=new_wts, strict=True)
  print("Weight loaded successfully!!")

  for i in range(len(image_path)):
    visualize_prediction(model=model,
                    image_path=image_path[i],
                    transform=transformation,
                    iou_threshold=iou_threshold,
                    anchors=anchors_list,
                    size=grid_size)


if __name__ == "__main__":
  paths = [imgs for imgs in os.listdir(arg.path) if imgs.endswith(".jpg") or imgs.endswith(".jpeg")]
  main(image_path=paths)
