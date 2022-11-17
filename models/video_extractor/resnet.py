import torch
import numpy as np
from torchvision.models import resnet34, ResNet34_Weights, resnet101, ResNet101_Weights

from .video_extractor import VideoExtractor

class ResNet34Extractor(VideoExtractor):
    def __init__(self, cfg):
        super(ResNet34Extractor, self).__init__(cfg)
        if cfg.pretrained:
            if not cfg.local_model:
                self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.model = resnet34()
                state_dict = torch.load(cfg.param_path)
                self.model.load_state_dict(state_dict)
        else:
            self.model = resnet34()
        
        self.cuda = cfg.cuda
        if cfg.cuda:
            self.model.cuda()

        self.model.eval()

    def forward(self, x: torch.Tensor):
        mu = torch.from_numpy(np.array(
            [0.485, 0.456,
            0.406])).float().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        std = torch.from_numpy(np.array(
            [0.229, 0.224,
            0.225])).float().unsqueeze(0).unsqueeze(2).unsqueeze(3)

        if self.cuda:
            x = x.cuda()
            mu = mu.cuda()
            std = std.cuda()

        x = (x - mu) / std
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet101Extractor(torch.nn.Module):
    def __init__(self, cfg):
        super(ResNet101Extractor, self).__init__()
        if cfg.pretrained:
            if not cfg.local_model:
                self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.model = resnet101()
                state_dict = torch.load(cfg.param_path)
                self.model.load_state_dict(state_dict)
        else:
            self.model = resnet101()
        
        self.cuda = cfg.cuda
        if cfg.cuda:
            self.model.cuda()

        self.model.eval()

    def forward(self, x):
        if self.cuda:
            x = x.cuda()

        mu = torch.from_numpy(np.array(
            [0.485, 0.456,
            0.406])).float().unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        std = torch.from_numpy(np.array(
            [0.229, 0.224,
            0.225])).float().unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        x = (x - mu) / std
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x