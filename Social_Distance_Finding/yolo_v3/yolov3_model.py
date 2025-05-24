import torch
from torch import nn
from yolo_v3.darknet_53 import Darknet53
from yolo_v3.conv_YOLOv3 import ConvLayer as convlayer

class YOLOv3(nn.Module):
  """
  This constructs the overall YOLOv3 architecture including Darknet53 backbone and after this another 53 convolutional layer to predict the bounding box co-ordinates in three stages.
  
  Returns the output of shape: [batch_size, grid_size, grid_size, number of anchor boxes, 5+class probabilities]
  """
  def __init__(self, input_size:int, anchors:list, num_classes:int):
    super(YOLOv3, self).__init__()
    self.num_anchors = len(anchors)
    self.num_classes = num_classes
    self.input_size = input_size
    self.num_coordinate = 1 + 4 + self.num_classes
    self.large_scale_downsample_factor = 32
    self.medium_scale_downsample_factor = 16
    self.small_scale_downsample_factor = 8
    self.large_scale_grid_size = self.input_size//self.large_scale_downsample_factor
    self.medium_scale_grid_size = self.input_size//self.medium_scale_downsample_factor
    self.small_scale_grid_size = self.input_size//self.small_scale_downsample_factor
    self.backbone = Darknet53()
    self.layer1 = nn.Sequential(
        convlayer(input_channels=1024, output_channels=512, kernel=1, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=512, output_channels=1024, kernel=3, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=1024, output_channels=512, kernel=1, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=512, output_channels=1024, kernel=3, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=1024, output_channels=512, kernel=1, activation_func="leakyrelu", stride=1)
    )
    self.stage1 = nn.Sequential(
        convlayer(input_channels=512, output_channels=1024, kernel=3, activation_func="leakyrelu", stride=1),
        nn.Conv2d(in_channels=1024, out_channels=self.num_anchors*(5+self.num_classes), kernel_size=1, stride=1)
    )
    self.upsample1 = nn.Sequential(
        convlayer(input_channels=512, output_channels=256, kernel=1, activation_func="leakyrelu", stride=1),
        nn.UpsamplingNearest2d(scale_factor=2)
    )
    self.layer2 = nn.Sequential(
        convlayer(input_channels=768, output_channels=256, kernel=1, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=256, output_channels=512, kernel=3, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=512, output_channels=256, kernel=1, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=256, output_channels=512, kernel=3, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=512, output_channels=256, kernel=1, activation_func="leakyrelu", stride=1)
    )
    self.stage2 = nn.Sequential(
        convlayer(input_channels=256, output_channels=512, kernel=3, activation_func="leakyrelu", stride=1),
        nn.Conv2d(in_channels=512, out_channels=self.num_anchors*(5+self.num_classes), kernel_size=1, stride=1)
    )
    self.upsample2 = nn.Sequential(
        convlayer(input_channels=256, output_channels=128, kernel=1, activation_func="leakyrelu", stride=1),
        nn.UpsamplingNearest2d(scale_factor=2)
    )
    self.layer3 = nn.Sequential(
        convlayer(input_channels=384, output_channels=128, kernel=1, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=128, output_channels=256, kernel=3, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=256, output_channels=128, kernel=1, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=128, output_channels=256, kernel=3, activation_func="leakyrelu", stride=1),
        convlayer(input_channels=256, output_channels=128, kernel=1, activation_func="leakyrelu", stride=1)
    )
    self.stage3 = nn.Sequential(
        convlayer(input_channels=128, output_channels=256, kernel=3, activation_func="leakyrelu", stride=1),
        nn.Conv2d(in_channels=256, out_channels=self.num_anchors*(5+self.num_classes), kernel_size=1, stride=1)
    )

  def forward(self, x:torch.Tensor):
    batch_size = x.shape[0]
    conv26, conv43, conv52 = self.backbone(x)
    layer1_out = self.layer1(conv52)
    stage1_out = self.stage1(layer1_out)
    upsample1_out = self.upsample1(layer1_out)
    combination1 = torch.cat([upsample1_out,conv43], dim=1)
    layer2_out = self.layer2(combination1)
    stage2_out = self.stage2(layer2_out)
    upsample2_out = self.upsample2(layer2_out)
    combination2 = torch.cat([upsample2_out, conv26], dim=1)
    layer3_out = self.layer3(combination2)
    stage3_out = self.stage3(layer3_out)

    stage1_out = stage1_out.contiguous().permute(0,2,3,1).view(batch_size, self.large_scale_grid_size, self.large_scale_grid_size, self.num_anchors, self.num_coordinate)
    
    stage1_x_center = stage1_out[..., 0:1]
    stage1_y_center = stage1_out[..., 1:2]
    stage1_width = stage1_out[..., 2:3]
    stage1_height = stage1_out[..., 3:4]
    stage1_confidence = stage1_out[..., 4:5]
    stage1_class = stage1_out[..., 5:]
    large_scale = torch.cat((stage1_x_center, stage1_y_center, stage1_width, stage1_height, stage1_confidence, stage1_class), dim=-1)

    stage2_out = stage2_out.contiguous().permute(0,2,3,1).view(batch_size, self.medium_scale_grid_size, self.medium_scale_grid_size, self.num_anchors, self.num_coordinate)
    
    stage2_x_center = stage2_out[..., 0:1]
    stage2_y_center = stage2_out[..., 1:2]
    stage2_width = stage2_out[..., 2:3]
    stage2_height = stage2_out[..., 3:4]
    stage2_confidence = stage2_out[..., 4:5]
    stage2_class = stage2_out[..., 5:]
    medium_scale = torch.cat((stage2_x_center, stage2_y_center, stage2_width, stage2_height, stage2_confidence, stage2_class), dim=-1)

    stage3_out = stage3_out.contiguous().permute(0,2,3,1).view(batch_size, self.small_scale_grid_size, self.small_scale_grid_size, self.num_anchors, self.num_coordinate)
    
    stage3_x_center = stage3_out[..., 0:1]
    stage3_y_center = stage3_out[..., 1:2]
    stage3_width = stage3_out[..., 2:3]
    stage3_height = stage3_out[..., 3:4]
    stage3_confidence = stage3_out[..., 4:5]
    stage3_class = stage3_out[..., 5:]
    small_scale = torch.cat((stage3_x_center, stage3_y_center, stage3_width, stage3_height, stage3_confidence, stage3_class), dim=-1)
    # print(f"shapes: {large_scale.shape}, {medium_scale.shape}, {small_scale.shape}")
    return large_scale, medium_scale, small_scale
    # return stage1_out, stage2_out, stage3_out

if __name__ == "__main__":
  anchors_list = [
            [116,90, 156,198, 373,326],
            [30,61, 62,45, 59,119],
            [10,13, 16,30, 33,23]]
  model = YOLOv3(input_size=256, anchors=anchors_list, num_classes=80)
  x = torch.randn((1,3,256,256))
  out1, out2, out3 = model(x)
  print(f"The shape of the outputs from the YOLOv3 model are: {out1.shape}, {out2.shape}, {out3.shape}")
  #Output:
  # The shape of the outputs from the YOLOv3 model are: torch.Size([1, 8, 8, 3, 85]), torch.Size([1, 16, 16, 3, 85]), torch.Size([1, 32, 32, 3, 85])
