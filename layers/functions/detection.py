#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch

from ..bbox_utils import decode, nms
from torch.autograd import Function


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(
            num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors,
                                       4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4),
                               batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(
                    boxes_, scores, self.nms_thresh, self.nms_top_k)
                count = count if count < self.top_k else self.top_k

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes_[ids[:count]]), 1)

        return output
        
class Detect1(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K_FACE
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        
        #st = time.time()
        conf_preds = conf_data.view(
            num, num_priors, self.num_classes).transpose(2, 1)
        #batch_priors = prior_data.view(-1, num_priors,
         #                              4).expand(num, num_priors, 4)
        #batch_priors = batch_priors.contiguous().view(-1, 4)

        #decoded_boxes = decode(loc_data.view(-1, 4),
        #                       batch_priors, self.variance)
        #decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output1 = torch.zeros(num,self.top_k, 6)
        #print("decoded_boxes:{}".format(time.time()-st))
        for i in range(num):
            boxes = decode(loc_data[i],prior_data,self.variance)
            conf_scores = conf_preds[i].clone()

            my_scores = []
            my_cat = []
            my_boxes = []
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                #print("c_mask shape:{}".format(c_mask.size()))
                scores = conf_scores[cl][c_mask]
                my_scores.append(scores)
                #my_cat.append(torch.ones_like(scores)*cl)
        
                if scores.dim() == 0:
                    continue
                #print(conf_scores[torch.arange(21).view(21,1),c_mask].size())
                my_cat.append(conf_scores[torch.arange(2).view(2,1),c_mask])
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes = boxes[l_mask].view(-1, 4)
                my_boxes.append(boxes)
                # idx of highest scoring and non-overlapping boxes per class
                if(boxes.size()[0]==0):
                    continue
                #ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                #output[i, cl, :count] = \
                #    torch.cat((scores[ids[:count]].unsqueeze(1),
                #               boxes[ids[:count]]), 1)
            my_scores = torch.cat(my_scores,0)
            my_cat = torch.cat(my_cat,1)
            my_boxes = torch.cat(my_boxes,0)
            my_cat = my_cat.transpose(0,1)
            #print(my_cat)
            #print("my_scores:{}".format(my_scores.size()))
            #print("my_boxes:{}".format(my_boxes.size()))
            #print("my_cat:{}".format(my_cat.size()))
            ids,count = nms(my_boxes,my_scores,self.nms_thresh,self.top_k)
            #print(my_scores[ids[:count]].size())
            #output1[i,:count] = torch.cat((my_cat[ids[:count]],my_scores[ids[:count]].unsqueeze(1),my_boxes[ids[:count]]),1)
            #print(my_cat[ids[:count]].size())
            #print(my_boxes[ids[:count]].size())
            output1[i,:count] = torch.cat((my_cat[ids[:count]],my_boxes[ids[:count]]),1)
        #print(count)
        return output1
