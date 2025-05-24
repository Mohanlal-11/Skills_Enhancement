import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
from PIL import Image
import random

class CocoImageFolder(Dataset):
  """
  This class is to make the custom ImageFolder to make ground truth as per the required by yolov3.

  Args:
    root(str): Path to the images (.jpg) and labels i.e. '.json' file
    image_size(int): Input size of image feed to model.
    grid_size(list): Size of feature maps from where the model makes predictions.
    transform : transformation function that performs transfromation on input images such as resizing, augmentation.
  """
  def __init__(self, root:str, image_size:int, grid_size:list=[8,16,32], transform=None):
    super(CocoImageFolder, self).__init__()
    self.image_size = image_size
    self.anchors = [
        [(116,90), (156,198), (373,326)],
        [(30,61), (62,45), (59,119)],
        [(10,13), (16,30), (33,23)]]
    self.grid_size = grid_size
    self.transform = transform
    self.num_prior_boxes = len(self.anchors)

    self.annotations = os.path.join(root, "labels.json") #This .json file contains a list that shows details of images and its annotataions i.e. "info", "licenses", "images", "annotations", "categories" but we need images, annotations and categories
    self.images = os.path.join(root, "data")

    with open(self.annotations, "r") as fc:
      self.coco_datasets = json.load(fc)

    self.images_list = [path for path in os.listdir(self.images) if path.endswith(".jpg")]
    self.images_dict = {img["id"]:img for img in self.coco_datasets["images"] if img["file_name"] in self.images_list}
    self.images_id = list(self.images_dict.keys())

    self.annotations_dict = {}
    for annot in self.coco_datasets["annotations"]:
      img_id = annot["image_id"]
      if img_id in self.images_dict.keys():
        if img_id not in self.annotations_dict:
          self.annotations_dict[img_id] = []
        self.annotations_dict[img_id].append(annot)

    self._original_classes_idx = {name["id"]:name["name"] for name in self.coco_datasets["categories"]}
    self.classes = {name:i for i,name in enumerate(self._original_classes_idx.values())}
    self.num_classes = len(self.classes)
    print(f"The total number of images and its annotations we have are: {len(self.images_dict)} and {len(self.annotations_dict)}")

  def __len__(self):
    return len(self.images_dict)

  def __getitem__(self, index):
    while True:
      image_id = self.images_id[index]
      image_informations = self.images_dict[image_id]
      image_name = image_informations["file_name"]
      image_path = os.path.join(self.images, image_name)

      image = Image.open(image_path)
      if image.mode == "L": #This is done because in coco dataset downloaded using fiftyoe tool, that may contains gray scale image i.e. one channel images, and our model takes only 3 channels images.
        index = (index+1)%len(self.images_id)
        continue
      transformation = transforms.Compose([
          transforms.Resize((self.image_size, self.image_size)),
          transforms.ToTensor()
      ])
      if self.transform:
        transformed_image = self.transform(image)
      else:
        transformed_image = transformation(image)

      ground_truth_bboxes = self.create_ground_truth(self.annotations_dict[image_id], image_informations["width"], image_informations["height"])

      return transformed_image, ground_truth_bboxes

  def create_ground_truth(self, annotations_list:list, width, height):
    ground_truth = [torch.zeros((S,S,self.num_prior_boxes,5+self.num_classes)) for S in self.grid_size]

    for annot in annotations_list:
      bbox = annot["bbox"]
      cat_id = annot["category_id"]
      cls = self._original_classes_idx[cat_id]
      label = self.classes[cls]

      x_min, y_min, wid, hei = bbox
      x_c = (x_min + wid / 2) / width
      y_c = (y_min + hei / 2) / height
      w = wid / width
      h = hei / height
      for scale_index, scale in enumerate(self.grid_size):
        grid_x = int(scale * x_c)
        grid_y = int(scale * y_c)

        grid_x, grid_y = min(grid_x, scale-1), min(grid_y, scale-1)
        x_center = (scale * x_c) - grid_x
        y_center = (scale * y_c) - grid_y
        w = scale * w
        h = scale * h

        box_coordinates = torch.tensor([1, x_center, y_center, w, h])

        ious = []
        for anchor in self.anchors[scale_index][:]:
          iou = self.calculate_iou_wh(anchor, (w, h))
          ious.append(iou)
        iou_tensor = torch.tensor(ious)
        best_anchor_idx = torch.argmax(iou_tensor, dim=0)

        ground_truth[scale_index][grid_x, grid_y, best_anchor_idx, 0:5] = box_coordinates
        ground_truth[scale_index][grid_x, grid_y, best_anchor_idx, 5+label] = 1

    return ground_truth

  def calculate_iou_wh(self, anchor_wh:tuple, ground_truth_wh:tuple):
    w1 = anchor_wh[0]
    h1 = anchor_wh[1]
    w2 = ground_truth_wh[0]
    h2 = ground_truth_wh[1]

    intersection = min(w1, w2) * min(h1, h2)
    union = (w1 * h1) + (w2 * h2) - intersection
    iou_wh = intersection/union
    return iou_wh

if __name__ == "__main__":
  data = CocoImageFolder(root="/content/coco_train",
                       image_size=256)
  classes = data.classes
  num_classes = len(classes)
  # print(f"The classes in ground truth are: {classes} ")
  x, g = data.__getitem__(1)

  print(f"the shape of ground truth lable of first index for large scale, medium scale and small scale are : {g[0].shape}, {g[1].shape}, {g[2].shape}")
  #Output:
  # the shape of ground truth lable of first index for large scale, medium scale and small scale are : torch.Size([8, 8, 3, 95]), torch.Size([16, 16, 3, 95]), torch.Size([32, 32, 3, 95])

