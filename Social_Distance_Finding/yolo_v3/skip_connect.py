import torch 
from torch import nn
from yolo_v3.conv_YOLOv3 import ConvLayer as convlayer

class Residual_Block(nn.Module):
  """
  This class is to perform the residual connection or skip connection as per the architectre of backbone of yolov3 i.e. Darknet53.
  """
  def __init__(self, channels:int):
    super(Residual_Block, self).__init__()

    self.skip_connection = nn.Sequential(
        convlayer(input_channels=2*channels, output_channels=channels, kernel=1),
        convlayer(input_channels=channels, output_channels=2*channels, kernel=3)
    )

  def forward(self, x:torch.Tensor):
    input_x = x
    res_out = self.skip_connection(x)
    res_out = res_out + input_x
    return res_out

if __name__ == "__main__":
  skip_conn = Residual_Block(channels=32)
  x = torch.randn((1,64,128,128))
  out = skip_conn(x)
  print(f"The shape of the input feature map before the residaul block or skip connection: {x.shape}") 
  print(f"The shape of the feature map after the residual block or skip connection : {out.shape}")
  #Outputs:
  # The shape of the input feature map before the residaul block or skip connection: torch.Size([1, 64, 128, 128])
  # The shape of the feature map after the residual block or skip connection : torch.Size([1, 64, 128, 128])
