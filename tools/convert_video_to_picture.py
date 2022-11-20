import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_dir(dirname):
    os.system(f'mkdir {dirname}/img')
    vnames = os.listdir(os.path.join(dirname, 'video'))
    for vname in tqdm(vnames):
        fullname = os.path.join(dirname, 'video', vname)
        extract_video(dirname, fullname, vname)

def extract_video(dirname, fullname, vname):
    cap = cv2.VideoCapture(fullname)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #! cnt means total frames

    step = int(cnt / 10)
    x = np.zeros((10, 224, 224, 3))
    cnt = 0
    i = 0

    while (cap.isOpened()):
        ret, frame = cap.read()  #! frame: [360, 640, 3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    x = np.average(x, 0) #! [10, 3, 224, 224] -> [3, 224, 224]
    cv2.imwrite(f'{dirname}/img/{vname}.png', x)
    

if __name__ == "__main__":
    extract_dir('/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Test/Noise')

