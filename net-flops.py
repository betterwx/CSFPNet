import torch
from CEPNet_Model import CEPNet
from thop import profile
import time
input = torch.randn(1, 3, 224, 224 )
time_start= time.time()
model = CEPNet(1)
model.load_state_dict(torch.load(r'D:\Paper\object detection paper\salient object detection\A论文代码\epnet2\epnet\epnet.pth.39'))
pred=model(input)
time_end=time.time()
time_sum=time_end-time_start
print('fps: ',1/time_sum)
flops, params = profile(model, inputs=(input, ))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))