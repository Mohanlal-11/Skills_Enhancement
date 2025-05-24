import fiftyone
from fiftyone import ViewField as F

def download_coco(number_of_sample:int, types:str="train", cls:list[str]=["cat", "dog"], dir:str="./coco"):
  """
  This function is to download the "coco" dataset of desired class and number of sample

  Arguments:
    number_of_sample(int): Argument for how many sample or datapoints we want.
    type(str): Argument for type of dataset we want which means "train" or "validation".
    cls(list[str]): Argument for how many classes want in dataset.
    dir(str): Argument for directory name where you want to export the dataset.

  Export the dataset of your requirements into the desired directory.
  """
  dataset = fiftyone.zoo.load_zoo_dataset(
      "coco-2017",
      split=types,
      classes=cls,
      shuffle=True,
      max_samples=number_of_sample,
  )
  # print(len(dataset.default_classes)) #This gives the total classes present in coco dataset
  view = dataset.filter_labels("ground_truth", F("label").is_in(cls))
  view.export(
      export_dir=f"{dir}_{types}",
      dataset_type=fiftyone.types.COCODetectionDataset,
  )
  fiftyone.launch_app(view) #This launches a app of fiftyone tool to visualize the downloaded dataset.
