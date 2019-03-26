import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import *
import os
from utils.viz import plot_bbox, plot_image
from matplotlib import pyplot as plt
from bnlstm import BNLSTM
import cv2
import time
from layers.modules.multibox_loss import MultiProjectLoss
from data.config import cfg

class association_lstm(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase,base, extras, head, num_classes):
        super(association_lstm, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        features_maps =[[75, 75], [38, 38], [19, 19], [9, 9], [5, 5], [3, 3]]
        self.priorbox = PriorBox([300,300], features_maps, cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.detect = Detect1(cfg)
        #self.MultiProjectLoss = MultiProjectLoss(self.num_classes,0,True,3,0.5)
        for p in self.parameters():
            p.requires_grad = False
        #self.roi_pool = _RoIPooling(self.cfg['POOLING_SIZE'], self.cfg['POOLING_SIZE'], 1.0 / 16.0)
        #self.roi_align = RoIAlignAvg(self.cfg['POOLING_SIZE'], self.cfg['POOLING_SIZE'], 1.0 / 16.0)

        #self.grid_size = self.cfg['POOLING_SIZE'] * 2 if self.cfg['CROP_RESIZE_WITH_MAX_POOL'] else self.cfg['POOLING_SIZE']
        #self.roi_crop = _RoICrop()
        self.img_shape = (300,300)
        self.tensor_len = 4+self.num_classes+49
        self.bnlstm1 = BNLSTM(input_size=55, hidden_size=150, batch_first=False, bidirectional=False)
        self.bnlstm2 = BNLSTM(input_size=150, hidden_size=300, batch_first=False, bidirectional=False)
        self.cls_pred = nn.Linear(300, self.num_classes)
        self.bbox_pred = nn.Linear(300, 4)
        self.association_pred = nn.Linear(300, 49)
        self.Con1_1 = nn.Conv2d(512,1,kernel_size=1,padding=0,dilation=1)
        self.scale = L2Norm(1,20)
        # if phase == 'vid_train':
        self.softmax = nn.Softmax(dim=-1)
        #     #self.detect = Trnsform_target(num_classes, 200, 0.5, 0.01, 0.45)
        #     self.detect = train_target(num_classes, 200, 0.5, 0.01, 0.45)
    
    def forward(self, x,h1=None,h2=None):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        batch_size = x.data.size(0)
        #print('input image size: ',x.size())
        display_img = x[0].clone().cpu().numpy().transpose((1,2,0))
        #print('display_img size: ',display_img.shape)
        
        for k in range(16):
            x = self.vgg[k](x)

        s = self.L2Norm3_3(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(16, 23):
            x = self.vgg[k](x)

        s = self.L2Norm4_3(x)
        sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)

        s = self.L2Norm5_3(x)
        sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        roi_feat = sources[2]
        roi_b,roi_c,roi_w,roi_h = roi_feat.size()
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        
        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        output = self.detect(
            loc.view(loc.size(0),-1,4),
            self.softmax(conf.view(conf.size(0),-1,self.num_classes)),
            self.priors.type(type(x.data))
            )
        return output
        m = nn.AdaptiveMaxPool2d((7,7))
        out = torch.zeros(batch_size,8,7,7)
        for b_it,b in enumerate(output):
            for n,s in enumerate(b):
                if(s[1]>0.5):
                    print(s)
                    x1,y1,x2,y2 = [x for x in s[-4:]]
                    x1,x2 = int(x1*roi_w),int(x2*roi_w)
                    y1,y2 = int(y1*roi_h),int(y2*roi_h)
                    print(x1,x2,y1,y2)
                    if(x1==x2):
                        x2 = x1+1
                    if(y1==y2):
                        y2 = y1+1
                    print(roi_feat[:,:,x1:x2,y1:y2].size())
                    print(m(roi_feat[:,:,x1:x2,y1:y2]).size())
                    out[b_it,n,:]=self.Con1_1(m(roi_feat[:,:,x1:x2,y1:y2]))
        
        out = self.scale(out).view(batch_size,8,-1) 
        print(out.size())
        out = torch.cat((output,out),2)
        print(out.size())
        if not h1:
            o1,h1 = self.bnlstm1(out)
        #print('output1 size: ',o1.size())
            o2,h2 = self.bnlstm2(o1)
        #print('output2 size: ',o2.size())
        else:
            o1,h1 = self.bnlstm1(out,h1)
            o2,h2 = self,bnlstm2(o1,h2)
        cls_pred = self.cls_pred(o2)
        print('cls_pred size: ',cls_pred.size())
        bbox_pred = self.bbox_pred(o2)
        print('bbox_pred size: ',bbox_pred.size())
        association_pred = self.association_pred(o2)
        print('association_pred size: ',association_pred.size())

        #loc_loss, cls_loss = self.MultiProjectLoss(cls_pred, bbox_pred, association_pred,output[:,:,-4:], output[:,:,1])
        #print('loc_loss size: ',loc_loss)
        #print('cls_loss size: ',cls_loss)
     #   pooled_feat = pooled_feat.view(pooled_feat.size(0), pooled_feat.size(1), -1)
        #print('output priors size: ',priors.size())
        #return bbox_pred, cls_pred, self.priors

        #print(self.priors.size())
        return output
        #return cls_pred,bbox_pred,association_pred,h1,h2,output
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage),strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]


def vgg(cfg, i, batch_norm=False):
    #s =time.time()
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    #print("VGG:{}".format(time.time()-s))
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, 28, -2]

    loc_layers += [nn.Conv2d(vgg[14].out_channels, 4,
                             kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(vgg[14].out_channels,
                              3 + (num_classes-1), kernel_size=3, padding=1)]

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


def build_association_lstm(phase, num_classes=2):
    base_, extras_, head_ = multibox(
        vgg(vgg_cfg, 3), add_extras((extras_cfg), 1024), num_classes)
    return association_lstm(phase, base_, extras_, head_, num_classes)
if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = build_association_lstm("test")
    model.load_weights('./weights/sfd_face_100000.pth')
    img_s = cv2.imread('10.jpg')
    h,w,c = img_s.shape
    print(img_s.shape)
    img = cv2.resize(img_s,(300,300))
    print(img.shape)
    img1 = torch.from_numpy(img).permute(2, 0, 1)
    input = img1.view(1,3,300,300).float()
    print(input.size())
    model = model.cuda(0)
    input = input.cuda(0)
    for i in range(10):
        start = time.time()
        output = model(input)
        print(time.time()-start)
    #cls_pred,bbox_pred = output
    #print(cls_pred.size())
    #print(bbox_pred.size())
    #print(association_pred.size())
    #print(output.size())
    for b in output:
        #for c in b[1:]:
        #if True:
            for s in b:
                if (s[1])>0.5:
                    print(s[1])
                    x1,y1,x2,y2 = [x for x in s[2:]]
                    x1,x2 = int(x1*w),int(x2*w)
                    y1,y2 = int(y1*h),int(y2*h)
                    cv2.rectangle(img_s,(x1,y1),(x2,y2),(0,255,0),1)
   # 
    cv2.imwrite("1.jpg",img_s)
