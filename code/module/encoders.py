import torch
import torch.nn as nn
from torch.nn import Module
from torchvision.models import ResNet
from module.basic import *
import module.resnet as resnet
import numpy as np

class PatchEncoder(Module):
    def __init__(self, layers=[2,2,2,2]):
        super(PatchEncoder, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1,padding=1, groups=1, bias=True, dilation=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, layers[0])
        self.layer2 = self._make_layer(64, layers[1], stride=2)
        self.layer3 = self._make_layer(128, layers[2], stride=2)
        self.layer4 = self._make_layer(256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        return x


# origine
class ImageEncoder(ResNet):
    def __init__(self,size=[540,540],resSet=[2,2,2,2],gar_latent_size=256, tran_mean=[0,0,0], infer_camera=False, light_instance_scale=1.0):
        super(ImageEncoder,self).__init__(BasicBlock,resSet)
        self.size=size
        self.avgpool=nn.AdaptiveAvgPool2d((8,8))
        self.fc=nn.Linear(512*64,2048)
        #origine
        self.dropout=nn.Dropout(p=0.5)
        #ft_overfit
        # self.dropout=nn.Dropout(p=0.9)
        self.shape_fc=nn.Linear(2048,10)
        self.pose_fc=nn.Linear(2048,24*9)	#output matrix formation
        self.tran_fc=nn.Linear(2048,3)
        self.tran_dp=nn.Dropout(p=0.3)
        self.gar_latent_size=gar_latent_size
        self.gar_Hierarchifs_size=64+128+256+512
        self.gar_fc=nn.Linear(2048,self.gar_latent_size)
        self.light_fc = nn.Linear(2048, 9*3)    # 预测光照
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.infer_camera = infer_camera
        if infer_camera:
            self.cam_fc = nn.Linear(2048, 3+3)
        #this mean value is from train set.
        # bcnet
        # self.register_buffer('tran_mean',torch.from_numpy(np.array([-1.0962e-02,  2.8778e-01,  1.2973e+01]).astype(np.float32)))
        # d3g
        # self.register_buffer('tran_mean',torch.from_numpy(np.array([4.1223e-03, 6.2957e-01, -5.6329e-04]).astype(np.float32)))
        # d3g + sizer
        # self.register_buffer('tran_mean',torch.from_numpy(np.array([0.0051,  0.4820, -0.0614]).astype(np.float32)))
        # auto
        self.register_buffer('tran_mean',torch.from_numpy(np.array(tran_mean).astype(np.float32)))
        self.light_instance_scale = light_instance_scale
        
    def forward(self,x):
        assert(x.shape[-2]==self.size[0])
        assert(x.shape[-1]==self.size[1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        fs1=x
        x = self.layer2(x)
        fs2=x
        x = self.layer3(x)
        fs3=x
        x = self.layer4(x)
        fs4=x
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x=self.fc(x)
        x = self.relu(x)
        shapes=self.shape_fc(x)
        poses=self.pose_fc(x)
        trans=self.tran_fc(self.tran_dp(x))+self.tran_mean.view(1,3)
        gars=self.gar_fc(x)
        lights = self.light_fc(x)
        # lights[:,3:] = self.sigmoid(lights[:,3:]) * self.light_instance_scale
        if self.infer_camera:
            cam_exts = self.cam_fc(x)
        if not self.infer_camera:
            return shapes,poses,trans,gars, lights, (fs1, fs2, fs3, fs4)
        else:
            return shapes,poses,trans,gars, cam_exts, lights, (fs1, fs2, fs3, fs4)
          
          
# for camera
class CameraEncoder(ResNet):
    def __init__(self,size=[540,540],resSet=[2,2,2,2]):
        super(CameraEncoder,self).__init__(BasicBlock,resSet)
        self.size=size
        self.avgpool=nn.AdaptiveAvgPool2d((8,8))
        self.fc=nn.Linear(512*64,2048)
        #origine
        # self.dropout=nn.Dropout(p=0.5)
        #ft_overfit
        self.dropout=nn.Dropout(p=0.9)
        self.cam_fc = nn.Linear(2048, 6)  # rx, ry, rz, tx, ty, tz
        
    def forward(self,x):
        assert(x.shape[-2]==self.size[0])
        assert(x.shape[-1]==self.size[1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        fs1=x
        x = self.layer2(x)
        fs2=x
        x = self.layer3(x)
        fs3=x
        x = self.layer4(x)
        fs4=x
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x=self.fc(x)
        x = self.relu(x)
        cam_exts = self.cam_fc(x) 
        return cam_exts
      
# for detail
class DetailEncoder(ResNet):
    def __init__(self,size=[540,540],resSet=[2,2,2,2], n_detail=128):
        super(DetailEncoder,self).__init__(BasicBlock,resSet)
        self.size=size
        self.avgpool=nn.AdaptiveAvgPool2d((8,8))
        self.fc=nn.Linear(512*64,2048)
        #origine
        # self.dropout=nn.Dropout(p=0.5)
        #ft_overfit
        self.dropout=nn.Dropout(p=0.9)
        self.detail_fc = nn.Linear(2048, n_detail)  # rx, ry, rz, tx, ty, tz
        
    def forward(self,x):
        assert(x.shape[-2]==self.size[0])
        assert(x.shape[-1]==self.size[1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        fs1=x
        x = self.layer2(x)
        fs2=x
        x = self.layer3(x)
        fs3=x
        x = self.layer4(x)
        fs4=x
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x=self.fc(x)
        x = self.relu(x)
        detail_latents = self.detail_fc(x) 
        return detail_latents
      
class ResnetEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters