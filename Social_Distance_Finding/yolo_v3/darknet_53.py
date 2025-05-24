import torch
from torch import nn
from yolo_v3.conv_YOLOv3 import ConvLayer as convlayer
from yolo_v3.skip_connect import Residual_Block as skipblock

class Darknet53(nn.Module):
  """
  This class constructs the backbone architecture i.e. Darknet53 of the YOLOv3 architecture.
  """
  def __init__(self):
    super(Darknet53,self).__init__()
    self.conv1 = convlayer(input_channels=3, output_channels=32, kernel=3, activation_func="leakyrelu", stride=1)
    self.conv2 = nn.Sequential(
        convlayer(input_channels=32, output_channels=64, kernel=3, activation_func="leakyrelu", stride=2),
        *[skipblock(channels=32) for i in range(1)]
    )
    self.conv3 = nn.Sequential(
        convlayer(input_channels=64, output_channels=128, kernel=3, activation_func="leakyrelu", stride=2),
        *[skipblock(channels=64) for i in range(2)]
    )
    self.conv4 = nn.Sequential(
        convlayer(input_channels=128, output_channels=256, kernel=3, activation_func="leakyrelu", stride=2),
        *[skipblock(channels=128) for i in range(8)]
    )
    self.conv5 = nn.Sequential(
        convlayer(input_channels=256, output_channels=512, kernel=3, activation_func="leakyrelu", stride=2),
        *[skipblock(channels=256) for i in range(8)]
    )
    self.conv6 = nn.Sequential(
        convlayer(input_channels=512, output_channels=1024, kernel=3, activation_func="leakyrelu", stride=2),
        *[skipblock(channels=512) for i in range(4)]
    )
    # Here, in *[skipblock() for i in range()], first it makes the list that contains the 'n' number(defined number in range()) of skip connection block and that ' * ' operator unpacks the element of that list.

  def forward(self, x:torch.Tensor):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out_1 = self.conv4(out)
    out_2 = self.conv5(out_1)
    out_3 = self.conv6(out_2)
    return out_1, out_2, out_3 # This Darknet53 returns the feature maps from 26th, 43th and last convolutional layer to obtain the fine-grained features. 

if __name__ == "__main__":
  darknet53 = Darknet53()
  x = torch.randn((1,3,256,256))
  out1, out2, out3 = darknet53(x)
  print(f"The shape of the output from the backbone is : {out1.shape}, {out2.shape}, {out3.shape}")
  #Output:
  # The shape of the output from the backbone is : torch.Size([1, 256, 32, 32]), torch.Size([1, 512, 16, 16]), torch.Size([1, 1024, 8, 8]) -----> This shapes are according to the original paper.
