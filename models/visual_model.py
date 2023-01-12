import torch 
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from transformers import ViTModel
import os

class MLP_Head(nn.Module):
    def __init__(self, ):
        super(MLP_Head, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.ReLU(),
        )
    
    def forward(self, feat):

        preds = self.head(feat)

        return preds




class IAANet(nn.Module):
    def __init__(self, args):
        super(IAANet, self).__init__()

        self.args = args

        if args.visual_model == 'vgg16':
            self.visual_base_model = VGG16(args)
        elif args.visual_model == 'resnet18':
            self.visual_base_model = ResNet18(args)
        elif args.visual_model == 'densenet121':
            self.visual_base_model = DenseNet121(args)
        elif args.visual_model == 'resnext50':
            self.visual_base_model = ResNext50(args)
        elif args.visual_model == 'vit':
            self.visual_base_model = Vit(args)



        self.head = MLP_Head()


    
    def forward(self, imgs, ):

  
        visual_feat = self.visual_base_model(imgs)

        preds = self.head(visual_feat)

        return preds


class Vit(nn.Module):
    def __init__(self, args):
        super(Vit, self).__init__()

        self.args = args
        self.backbone = ViTModel.from_pretrained(os.path.join(args.huggingface_model_root, 'vit-base-patch16-224-in21k'))
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, imgs):
    
        vit_output = self.backbone(pixel_values=imgs)
        feat = vit_output.last_hidden_state[:, 0]

        feat = self.activate(feat)

        # feat = vit_output.pooler_output

        return feat


class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.args = args
        self.backbone = models.vgg16(pretrained=True)
        self.backbone.classifier = self.backbone.classifier[:5]

        self.head = nn.Sequential(
            nn.Linear(4096, 768),
            nn.LeakyReLU(0.2),

        )
    
    def forward(self, imgs):
    
        feat = self.backbone(imgs)

        feat = self.head(feat)

        return feat

class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18, self).__init__()
        self.args = args
        self.backbone = models.resnet18(pretrained=True)

        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 768),
            nn.LeakyReLU(0.2),

        )

    def forward(self, imgs):

        feat = self.backbone(imgs)

        return feat


class ResNext50(nn.Module):
    def __init__(self, args):
        super(ResNext50, self).__init__()
        self.args = args
        self.backbone = models.resnext50_32x4d(pretrained=True)

        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 768),
            nn.LeakyReLU(0.2),
        )

    def forward(self, imgs):

        feat = self.backbone(imgs)

        return feat



class DenseNet121(nn.Module):
    def __init__(self, args):
        super(DenseNet121, self).__init__()
        self.args = args
        self.backbone = models.densenet121(pretrained=True)

        self.backbone.classifier = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LeakyReLU(0.2),
        )

    def forward(self, imgs):

        feat = self.backbone(imgs)

        return feat




