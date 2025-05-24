import torch
from torch import nn

class ConvLayer(nn.Module):
  """
  This is to construct the convolutional layer required for the backbone of the yolov3 i.e.Darknet53 and for the detection.
  """
  def __init__(self, input_channels:int, output_channels:int, kernel:int, activation_func:str="leakyrelu", stride:int=1):
    super(ConvLayer, self).__init__()
    
    if activation_func == "relu":
      act_fn = nn.ReLU(inplace=True)
    elif activation_func == "leakyrelu":
      act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True) # In case of activation function used the default is "LeakyReLU"
    
    c_padding = (kernel-1)//2 if stride == 1 else 0 # This padding is 0 when the stride is '2' which means it will be 0 when the feature map's size has to be reduced. 
    self.stride = stride
    self.s_padding = nn.ZeroPad2d((1,0,1,0)) # This padding adds 0's of one column and one row to increase the width and height of the input feature map by 1 pixel so that the feature map wil be reduces to its half. E.g. if input featuer map is of [1,64,32,32] then after this padding it will be [1,64,33,33]
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel, stride=stride, padding=c_padding, bias=False),
        nn.BatchNorm2d(output_channels),
        act_fn
    )
  def forward(self, x:torch.Tensor):
    out = self.conv(self.s_padding(x)) if self.stride > 1 else self.conv(x)
    return out

if __name__ == "__main__":
  x = torch.randn((1,3,416,416))
  stride=2
  convlayer = ConvLayer(input_channels=3, output_channels=64, kernel=3, activation_func="leakyrelu", stride=stride)
  output = convlayer(x)
  print(f"The shape of input to the convlayer is : {x.shape}")
  print(f"The shape of output from the convlayer class is when stride is {stride} : {output.shape}")
  print(f"The shape of input after applying 'nn.ZeroPad2d((1,0,1,0))' : {convlayer.s_padding(x).shape}")
  #Outputs: 
  # The shape of input to the convlayer is : torch.Size([1, 3, 416, 416])
  # The shape of output from the convlayer class is when stride is 1 : torch.Size([1, 64, 416, 416])
  # The shape of output from the convlayer class is when stride is 2 : torch.Size([1, 64, 208, 208])
  # The shape of input after applying 'nn.ZeroPad2d((1,0,1,0))' : torch.Size([1, 3, 417, 417])
