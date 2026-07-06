import math

import torch
import torch.nn as nn

from utils.util import make_anchors

import torchvision


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))




"""
Change with your own backbone
You must extract 3 semantic level of information from your network : P3, P4 and P5
"""
class YourNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small()
        self.features = nn.ModuleList([backbone.features[:4], backbone.features[4:9], backbone.features[9:-1]])
    def forward(self, x):
        res=[]
        for f in self.features:
            x = f(x)
            res.append(x)
        return res

    
def activation_function(act="RE"):
    res = nn.Hardswish()
    if act == "RE":
        res = nn.ReLU6(inplace=True)
    elif act == "GE":
        res = nn.GELU()
    elif act == "SI":
        res = nn.SiLU()
    elif act == "EL":
        res = nn.ELU()
    else:
        res = nn.Hardswish()
    return res


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act="RE", p=None, g=1, d=1):
        super().__init__()
        padding = 0 if k==s else autopad(k,p,d)
        self.c = nn.Conv2d(c1, c2, k, s, padding, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU()
        #self.act = activation_function(act)
    
    def forward(self, x):
        return self.act(self.bn(self.c(x)))


class MobileNetV3_BLOCK(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, act="GE", stride=1, pw=True):
        super().__init__()

        #act = nn.ReLU6(inplace=True) if NL=="RE" else nn.Hardswish()
        c_mid = e if e != None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else [] #if c_mid != c1 else []
        features.extend([mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                         nn.Conv2d(c_mid, c2, 1),
                         nn.BatchNorm2d(c2),
                         ])
        self.layers = nn.Sequential(*features)
    def forward(self, x):
        #print(x.shape)
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)
        

class LeNeck(torch.nn.Module):
    def __init__(self, width, multiplier=1.0, depth=1.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        
        self.p4_up = [MobileNetV3_BLOCK(c1=width[1]+width[2], c2=int(64*multiplier), e=int(128*multiplier), k=5)]
        self.p4_up += [MobileNetV3_BLOCK(c1=int(64*multiplier), c2=int(64*multiplier), e=int(128*multiplier), k=5) for _ in range(int(2*depth))]
        self.p4_up = nn.Sequential(*self.p4_up)

        self.p3_up = [MobileNetV3_BLOCK(c1=int(64*multiplier) + width[0], c2=int(32*multiplier), e=int(64*multiplier)+width[0], k=3, pw=False)]
        self.p3_up += [MobileNetV3_BLOCK(c1=int(32*multiplier), c2=int(32*multiplier), e=int(96*multiplier), k=3) for _ in range(int(2*depth))]
        self.p3_up = nn.Sequential(*self.p3_up)

        self.p3_downsampling = mn_conv(int(32*multiplier), int(64*multiplier), k=3, s=2, p=1)

        self.p4_down = [MobileNetV3_BLOCK(c1=int(64*multiplier)+int(64*multiplier), c2=int(64*multiplier), e=int(128*multiplier), k=5)]
        self.p4_down += [MobileNetV3_BLOCK(c1=int(64*multiplier), c2=int(64*multiplier), e=int(128*multiplier), k=5) for _ in range(int(2*depth))]
        self.p4_down = nn.Sequential(*self.p4_down)
        self.p4_downsampling = mn_conv(int(64*multiplier), int(96*multiplier), k=3, s=2, p=1)

        self.p5_down = [MobileNetV3_BLOCK(c1=int(96*multiplier)+width[2], c2=int(96*multiplier), e=int(96*multiplier)+width[2], k=5)]
        self.p5_down += [MobileNetV3_BLOCK(c1=int(96*multiplier), c2=int(96*multiplier), e=int(192*multiplier), k=5) for _ in range(int(2*depth))]
        self.p5_down = nn.Sequential(*self.p5_down)

        
    
    def forward(self, x):
        p3,p4,p5 = x
        
        p5_up = torch.cat(tensors=[self.up(p5), p4], dim=1)    #P5 ----> to P4 (no computation at P5 in the down-up passage)
                                                #P4_UP concat with P4_Backbone
        p4_up = self.p4_up(p5_up)               #P4_UP conv computations  (save for later, P4 top-down passage)

        p3_up = torch.cat(tensors=[self.up(p4_up), p3], dim=1) #P4_UP ---> P3 and concat with P3 backbone
        p3_up = self.p3_up(p3_up)               #P3 computation (save for later, no computation at P3 top-down passage, NECK OUTPUT)
        p4_down = self.p3_downsampling(p3_up)   #P3_UP ---> P4_DOWN
        
        p4_down = torch.cat(tensors=[p4_down, p4_up], dim=1)   #Concat with P4_UP from the down-up passage
        p4_down = self.p4_down(p4_down)         #Computations (save for later, NECK output)
        
        p5_down = self.p4_downsampling(p4_down) #p4_DOWN --> P5_DOWN (and last neck computation)
        p5_down = torch.cat(tensors=[p5_down, p5], dim=1)
        p5_down = self.p5_down(p5_down)         #P5_DOWN Computation (save for NECK OUTPUT)

        return p3_up, p4_down, p5_down
    



class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)




class LeHead(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box, k=1),
                                                           Conv(box, box, k=3, p=1, g=box),
                                                           Conv(box, box, k=3, p=1, g=box),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, cls, k=1),
                                                           Conv(cls, cls, k=3, p=1, g=cls),
                                                           Conv(cls, cls, k=3, p=1, g=cls),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)



class LeYOLO(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        multiplier=1.5
        depth=2
        self.net = YourNet()
        width=[24, 48, 96] #improve by walking through the backbone and extract the exact number of channels
        #special width
        self.fpn = LeNeck(width, multiplier=multiplier, depth=depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = LeHead(num_classes, (int(32*multiplier), int(64*multiplier), int(96*multiplier))) #lEYOLO Outputs channels (which is not the same from DarkFPN and/or neurocorgi backbone outputs)
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def leyolo_n(num_classes: int = 80):
    return LeYOLO(num_classes)
