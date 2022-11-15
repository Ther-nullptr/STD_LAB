import os
from torchvision.models import resnet34, ResNet34_Weights
import cv2
import torch
import numpy as np

def _model(pretrained = False):
    if pretrained:
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = resnet34()
    model.cuda()
    model.eval()
    return model


def extract_feature(x, model):
    mu = torch.from_numpy(np.array(
        [0.485, 0.456,
         0.406])).float().unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
    std = torch.from_numpy(np.array(
        [0.229, 0.224,
         0.225])).float().unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
    x = (x - mu) / std
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def extract_dir(model, dirname):
    os.system('mkdir {}/vfeat2048'.format(dirname))
    vnames = os.listdir(os.path.join(dirname, 'video'))
    for vname in vnames:
        sname = vname[:-4] + '.npy'
        fullname = os.path.join(dirname, 'video', vname)
        feat = extract_video(fullname, model)
        np.save(os.path.join(dirname, 'vfeat2048', sname), feat)


def extract_video(fullname, model=_model()):
    cap = cv2.VideoCapture(fullname)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #! cnt means total frames

    step = int(cnt / 10)
    x = np.zeros((10, 224, 224, 3))
    cnt = 0
    i = 0

    while (cap.isOpened()):
        ret, frame = cap.read()  #! frame: [360, 640, 3]
        if not ret:
            break

        if cnt == 0:
            h = frame.shape[0]
            w = frame.shape[1]
            if h > w:
                news = (int(w / h * 224), 224)
            else:
                news = (224, int(h / w * 224)) #! resize to fix value (16/9) [224, 126]

        if cnt == step * i:
            if h > w:
                x[i, :, int((224 - news[0]) / 2):int((224 - news[0]) / 2) + news[0]] = cv2.resize(frame, news)[:, :, ::-1].copy() #! 1. resize the frame 2. reverse the channel 3. put the image to the middle
            else:
                x[i, int((224 - news[1]) / 2):int((224 - news[1]) / 2) + news[1], :] = cv2.resize(frame, news)[:, :, ::-1].copy()
            i += 1
        if i == 10:
            break
        cnt += 1

    cap.release()
    x = torch.from_numpy(x / 255.0).float().cuda().permute(0, 3, 1, 2) #! [10, 3, 224, 224]
    with torch.no_grad():
        feat = extract_feature(x, model).cpu().numpy()
    print(feat.shape) #! [10, 512]
    return feat


if __name__ == '__main__':
    extract_dir('Train')
    extract_dir('Test/Clean')
    extract_dir('Test/Noise')
