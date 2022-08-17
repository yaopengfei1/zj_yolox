
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from loguru import logger
#from detectron2.modeling.backbone import Backbone
#from detectron2.layers import ShapeSpec

#__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b']

model_urls = {
    'res2net50_v1b_26w_4s':
    'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s':
    'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=26,
                 scale=4,
                 stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes,
                               width * scale,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width,
                          width,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 baseWidth=26,
                 scale=4,
                 num_classes=None,
                 out_features=("res3", "res4", "res5"),
                 color_channel=3,
                 freeze=False,
                 image_net_pre_train=True,
                 pretrain_path=None):
        self.inplanes = 64
        super(Res2Net, self).__init__()

        self._out_feature_strides = {"stem": 32}
        self._out_feature_channels = {"stem": 64}
        self._out_features = out_features
        self.num_classes = num_classes

        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv2d(color_channel, 32, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self._out_feature_channels['res3'] = 128 * block.expansion
        self._out_feature_channels['res4'] = 256 * block.expansion
        self._out_feature_channels['res5'] = 512 * block.expansion
        self._out_feature_strides['res3'] = 8
        self._out_feature_strides['res4'] = 16
        self._out_feature_strides['res5'] = 32

        if num_classes:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outputs['stem'] = x

        x = self.layer1(x)
        x = self.layer2(x)
        outputs['res3'] = x
        x = self.layer3(x)
        outputs['res4'] = x
        x = self.layer4(x)
        outputs['res5'] = x

        if self.num_classes:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            #return outputs
            return {
                k: v
                for k, v in outputs.items() if k in self._out_features
            }

    def output_shape(self):
        return {k: outputs[k].size() for k in outputs}

    @property
    def size_divisibility(self) -> int:
        return 32


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['res2net50_v1b_26w_4s']),
                              strict=False)
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3],
                    baseWidth=26,
                    scale=4,
                    **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['res2net101_v1b_26w_4s']),
                              strict=False)
    return model


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['res2net50_v1b_26w_4s']),
                              strict=False)
    return model


def res2net50_v1b_26w_csp(pretrained: bool = False,
                          depth: float = 0.75,
                          expansion: int = 1,
                          **kwargs) -> Res2Net:
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    image_net=kwargs['image_net_pre_train']
    freeze=kwargs['freeze']
    pretrain_path=kwargs['pretrain_path']
    # logger.info(f'-----------------{image_net}_{freeze}')
    Bottle2neck.expansion = expansion
    blocks = [int(i * depth) for i in [3, 4, 6, 3]]
    model = Res2Net(Bottle2neck, blocks, baseWidth=26, scale=4, **kwargs)
    if pretrained:
        net_weights = model.state_dict()
        pre_weights=None
        if image_net:
            logger.info('res2net50_v1b_26w_csp load [image-net] weight')
            pre_weights = model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'])
        else:
            logger.info('res2net50_v1b_26w_csp load [classify ] weight')
            model_weights_path = pretrain_path
            pre_weights = torch.load(model_weights_path)['model']
        new_dict={}
        for k,v in pre_weights.items():
            if not net_weights.keys().__contains__(k):
                continue
            if net_weights[k].numel() == v.numel():
                new_dict[k]=v        
        
        logger.info(f'res2net50_v1b_26w_csp load dict,  match state dict count: {len(new_dict)}')
        model.load_state_dict(new_dict,strict=False)#missing count:80  load_dict:286-80
        
        
        #model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']),strict=False)
    
    if freeze:
        logger.info('res2net50_v1b_26w_csp [freeze]')
        for p in model.parameters():
            p.requires_grad = False
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3],
                    baseWidth=26,
                    scale=4,
                    **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['res2net101_v1b_26w_4s']),
                              strict=False)
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 8, 36, 3],
                    baseWidth=26,
                    scale=4,
                    **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['res2net152_v1b_26w_4s']),
                              strict=False)
    return model


if __name__ == '__main__':    
    save_path='/data/lsc/zj_yolox/state_dict-epoch-2.pth'
    net12=res2net50_v1b_26w_csp(depth=0.75)
    net_weights=net12.state_dict()
    
    pre_weights=torch.load(save_path)
    new_dict={}
    for k,v in pre_weights.items():
        if not net_weights.keys().__contains__(k):
            continue
        
        if net_weights[k].numel() == v.numel():
            new_dict[k]=v
    print(len(new_dict))
    print(len(pre_weights.keys()))
    
    # net12.load_state_dict(torch.load(save_path))
    # print(net12)
    exit()
    
    torch.save(net12, '/data/lsc/zj_yolox/res2net---1.pth') 
    
    
    net = res2net50_v1b_26w_csp(depth=0.75)
    net_weights=net.state_dict()
    
    
    net12_weights=torch.load(save_path).state_dict()
    
    new_dict={}
    for k,v in net12_weights.items():
        if not net_weights.keys().__contains__(k):
            continue
        
        if net_weights[k].numel() == v.numel():
            new_dict[k]=v
    print(len(new_dict))
    
    net.load_state_dict(new_dict, strict=True)
    exit()
    
    net.load_state_dict(new_dict, strict=False)
    #net.load_state_dict(new_dict)
    
    
    net.load_state_dict(net12_weights, strict=True)
    
    exit()
    
    
    net_weights = net.state_dict()
    model_weights_path = "/data/lsc/zj_yolox/res2net-0.956.pth.tar"
    pre_weights = torch.load(model_weights_path)['model']
    
    print(net)
    print('net_weights----------------------------')
    print(net_weights.keys())
    exit()
    
    new_dict={}
    for k,v in pre_weights.items():
        if not net_weights.keys().__contains__(k):
            continue
        
        if net_weights[k].numel() == v.numel():
            new_dict[k]=v
    print(len(new_dict))
    print(len(pre_weights.keys()))
    net.load_state_dict(new_dict, strict=False)
    #net.load_state_dict(new_dict)
    exit()
    

    print('--------------',len(pre_dict))
    net.load_state_dict(pre_dict, strict=False)
    exit()


    images = torch.rand(1, 3, 640, 640).cuda(0)
    model = res2net50_v1b_26w_csp(pretrained=True)
    model = model.cuda(0)
    outputs = model(images)
    print([(k, outputs[k].size()) for k in outputs])
    print(model.output_shape())
    #for name, parameters in model.named_parameters():
    #    print(name, ':', parameters.size())
