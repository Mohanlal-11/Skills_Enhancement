import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast #Automatic Mixed Precision Package
from utils import calculate_iou, nms, mAP, save_weights, transform_predicted_txtytwth
from yolov3_model import YOLOv3
from yolov3imagefolder import CocoImageFolder
from yolov3lossfunc import Yolov3Loss
import argparse
import os
from tqdm.auto import tqdm

"""
This script is to train the yolov3 architecture written in pytorch from scratch on coco dataset. 
And saves the weights of trained model into a particular directory. 
"""

parser = argparse.ArgumentParser(description="Parser for taking the required arguments for the training of yolov3 from scratch.")
parser.add_argument("-e", "--epochs", type=int, default=20, help="Argument for how many times there will be forward propagation.")
parser.add_argument("-b", "--batch", type=int, default=32, help="Argument for how many samples will there be in one chunk of data during training")
parser.add_argument("-d", "--dir", type=str, default="./Yolov3_weights", help="Argumnet fot where we want to save our trained weights.")
parser.add_argument("-n", "--name", type=str, default="Yolov3_weights_coco_2017.pt", help="Argument for the name of trained weights.")
parser.add_argument("-i", "--input_size", type=int, default=256, help="Argument for the desired input size to model.")
parser.add_argument("-p", "--data_path", type=str, default="./coco_train", help="Argument for the training dataset path")
parser.add_argument("-y", "--save_bool", type=bool, default=False, help="Argument for whether the weights are saving or not.")

arg = parser.parse_args()

def train_step(model:torch.nn.Module,
               train_dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               LR:float,
               device:torch.device):
  scaler = GradScaler()
  model = model.to(device)
  train_loss = 0
  optimizer = torch.optim.SGD(params=model.parameters(),
                              lr=LR,
                              momentum=0.9,
                              weight_decay=0.0005)
  # optimizer = torch.optim.Adam(params=model.parameters(),lr=LR)

  model.train()
  for batch, (image, label) in enumerate(train_dataloader):
    image, label[0], label[1], label[2] = image.to(device), label[0].to(device), label[1].to(device), label[2].to(device)

    with autocast():
      large_scale_prediction, medium_scale_prediction, small_scale_prediction = model(image)
      loss = (
          loss_fn(predictions=large_scale_prediction, labels=label[0], scale="large", device=device)
          + loss_fn(predictions=medium_scale_prediction, labels=label[1], scale="medium", device=device)
          + loss_fn(predictions=small_scale_prediction, labels=label[2], scale="small", device=device)
      )
    train_loss += loss.item()

    optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

  train_loss /= (3*len(train_dataloader))
  return train_loss

def validation_step(model:torch.nn.Module,
                    valid_dataloader:torch.utils.data.DataLoader,
                    loss_fn:torch.nn.Module,
                    num_class:int,
                    grid_size:list,
                    device:torch.device):
  model = model.to(device)
  valid_loss = 0
  best_map_large, best_map_medium, best_map_small = 0, 0, 0
  model.eval()
  with torch.inference_mode():
    for batch, (image, label) in enumerate(valid_dataloader):
      image, label[0], label[1], label[2] = image.to(device), label[0].to(device), label[1].to(device), label[2].to(device)
      with autocast():
        large_scale_prediction, medium_scale_prediction, small_scale_prediction = model(image)

        loss = (
            loss_fn(predictions=large_scale_prediction, labels=label[0], scale="large", device=device)
            + loss_fn(predictions=medium_scale_prediction, labels=label[1], scale="medium", device=device)
            + loss_fn(predictions=small_scale_prediction, labels=label[2], scale="small", device=device)
        )
        valid_loss += loss.item()

      # large_scale_prediction = transform_predicted_txtytwth(bboxes=large_scale_prediction, grid_size=grid_size, device=device, scale="large")
      # medium_scale_prediction = transform_predicted_txtytwth(bboxes=medium_scale_prediction, grid_size=grid_size, device=device, scale="medium")
      # small_scale_prediction = transform_predicted_txtytwth(bboxes=small_scale_prediction, grid_size=grid_size, device=device, scale="small")

      # label[0] = transform_predicted_txtytwth(bboxes=label[0], grid_size=grid_size, device=device, scale="large")
      # label[1] = transform_predicted_txtytwth(bboxes=label[1], grid_size=grid_size, device=device, scale="medium")
      # label[2] = transform_predicted_txtytwth(bboxes=label[2], grid_size=grid_size, device=device, scale="small")

      # large_scale_prediction = large_scale_prediction.flatten(0,3)
      # medium_scale_prediction = medium_scale_prediction.flatten(0,3)
      # small_scale_prediction = small_scale_prediction.flatten(0,3)

      # selected_large_scale = nms(pred_bboxes=large_scale_prediction.tolist(),prob_threshold=0.60, iou_threshold=0.3, format="center")
      # selected_medium_scale = nms(pred_bboxes=medium_scale_prediction.tolist(), prob_threshold=0.60, iou_threshold=0.3, format="center")
      # selected_small_scale = nms(pred_bboxes=small_scale_prediction.tolist(), prob_threshold=0.60, iou_threshold=0.3, format="center")

      # label[0] = label[0].flatten(0,3)
      # label[1] = label[1].flatten(0,3)
      # label[2] = label[2].flatten(0,3)

      # label[0] = label[0].tolist()
      # label[1] = label[1].tolist()
      # label[2] = label[2].tolist()

      # map_large = mAP(pred_bboxes=selected_large_scale,
      #                 ground_truth_bboxes=label[0],
      #                 iou_threshold=0.65,
      #                 format="center",
      #                 num_classes=num_class)
      # if best_map_large < map_large:
      #   best_map_large = map_large

      # map_medium = mAP(pred_bboxes=selected_medium_scale,
      #                 ground_truth_bboxes=label[1],
      #                 iou_threshold=0.65,
      #                 format="center",
      #                 num_classes=num_class)
      # if best_map_medium < map_medium:
      #   best_map_medium = map_medium

      # map_small = mAP(pred_bboxes=selected_small_scale,
      #                 ground_truth_bboxes=label[2],
      #                 iou_threshold=0.65,
      #                 format="center",
      #                 num_classes=num_class)
      # if best_map_small < map_small:
      #   best_map_small = map_small        #--------------> These commented lines are for the calculation of mean average precision of the yolov3 
    valid_loss /= (3*len(valid_dataloader))
  return valid_loss, best_map_large, best_map_medium, best_map_small

def train_function(model:torch.nn.Module,
          train_loader:torch.utils.data.DataLoader,
          valid_loader:torch.utils.data.DataLoader,
          epochs:int,
          loss_fn:torch.nn.Module,
          device:torch.device,
          num_class:int,
          grid_size:list):
  results = {
      "train_loss": [],
      "valid_loss": [],
      "large_scale_best_mAP": [],
      "medium_scale_best_mAP": [],
      "small_scale_best_mAP": []
  }
  lr = 0.001
  for epoch in tqdm(range(epochs)):
    if epoch == 70 or epoch == 120:
      lr = lr/10
    train_loss = train_step(model=model,
                            train_dataloader=train_loader,
                            loss_fn=loss_fn,
                            LR=lr,
                            device=device)
    valid_loss, large_map, medium_map, small_map = validation_step(model=model,
                                                                   valid_dataloader=valid_loader,
                                                                   loss_fn=loss_fn,
                                                                   num_class=num_class,
                                                                   grid_size=grid_size,
                                                                   device=device)

    print(
        f"Train_loss: {train_loss} | "
        f"Valid_loss: {valid_loss} | "
        f"Large Scale mAP: {large_map} | "
        f"Medium Scale mAP: {medium_map} | "
        f"Small Scale mAP: {small_map} | "
    )
    results["train_loss"].append(train_loss)
    results["valid_loss"].append(valid_loss)
    results["large_scale_best_mAP"].append(large_map)
    results["medium_scale_best_mAP"].append(medium_map)
    results["small_scale_best_mAP"].append(small_map)
  return results

def dataloader_func(dataset:torch.utils.data.Dataset,
                    batch_size:int,
                    shuffle:bool):
  dataloader = DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=os.cpu_count()
  )
  return dataloader

def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  anchors_list = [
            [116,90, 156,198, 373,326],
            [30,61, 62,45, 59,119],
            [10,13, 16,30, 33,23]]

  grid_size = [arg.input_size//32, arg.input_size//16, arg.input_size//8]

  train_dataset_imgfolder = CocoImageFolder(root=arg.data_path, image_size=arg.input_size, grid_size=grid_size)

  valid_dataset_imgfolder = CocoImageFolder(root=arg.data_path, image_size=arg.input_size, grid_size=grid_size)

  train_dataloader = dataloader_func(dataset=train_dataset_imgfolder,
                                     batch_size=arg.batch,
                                     shuffle=True)
  valid_dataloader = dataloader_func(dataset=valid_dataset_imgfolder,
                                     batch_size=arg.batch,
                                     shuffle=True)

  classes = train_dataset_imgfolder.classes
  num_class = train_dataset_imgfolder.num_classes
  print(f"[INFO] The number of classes in dataset is: {num_class} and classe are: {classes}")
  print(f"[INFO] The total dataset in train and valid dataset are: {train_dataset_imgfolder.__len__()} and {valid_dataset_imgfolder.__len__()}")

  model_yolov3 = YOLOv3(input_size=arg.input_size, anchors=anchors_list, num_classes=num_class)
  loss_function = Yolov3Loss(anchors=anchors_list, input_size=arg.input_size)


  train_function(
      model=model_yolov3,
      train_loader=train_dataloader,
      valid_loader=valid_dataloader,
      epochs=arg.epochs,
      loss_fn=loss_function,
      device=device,
      num_class=num_class,
      grid_size=grid_size
  )
  print(f"[INFO] Model training is finished.................!!")
  if arg.save_bool:
    save_weights(
        model=model_yolov3,
        save_dir=arg.dir,
        model_name=arg.name
    )

if __name__ == "__main__":
  main()
