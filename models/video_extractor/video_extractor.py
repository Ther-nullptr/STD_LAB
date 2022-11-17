import os
import cv2
import torch
import numpy as np
from abc import abstractmethod

class VideoExtractor(torch.nn.Module):
    def __init__(self, cfg):
        super(VideoExtractor, self).__init__()
        self.model = torch.nn.Module()
        self.name = cfg.model

    @abstractmethod
    def forward(self, x):
        pass

    def extract_dir(self, dirname):
        os.system(f'mkdir {dirname}/{self.name}')
        vnames = os.listdir(os.path.join(dirname, 'video'))
        for vname in vnames:
            sname = vname[:-4] + '.npy'
            fullname = os.path.join(dirname, 'video', vname)
            feat = self.extract_video(fullname, self.model)
            np.save(os.path.join(dirname, self.name, sname), feat)

    def extract_video(self, fullname):
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
        x = torch.from_numpy(x / 255.0).float().permute(0, 3, 1, 2) #! [10, 3, 224, 224]
        with torch.no_grad():
            feat = self.model(x).cpu().numpy()

        return feat