import cv2
import numpy as np

def extract_video(name):
    cap = cv2.VideoCapture(name)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #! cnt means total frames

    step = int(cnt / 10)
    cnt = 0
    i = 0

    frame_list = []

    while (cap.isOpened()):
        ret, frame = cap.read()  #! frame: [360, 640, 3]
        if not ret:
            break

        if cnt == step * i:
            frame_list.append(frame[:,:,::-1])
            i += 1

        if i == 10:
            break

        cnt += 1

    cap.release()
    return np.stack(frame_list)

if __name__ == '__main__':
    pictures_clean = extract_video('/root/kyzhang/yjwang/InclusiveFL2/test/noise/clean_video_2.mp4')
    pictures_noise = extract_video('/root/kyzhang/yjwang/InclusiveFL2/test/noise/noise_video_2.mp4')
    noise = (pictures_noise.astype('int8') - pictures_clean.astype('int8'))
    print(np.mean(np.abs(pictures_noise.astype('int8') - pictures_clean.astype('int8'))))
    
    # plot the distribution of noise
    import matplotlib.pyplot as plt
    plt.hist(noise[5].flatten(), bins=100)
    from scipy.stats import norm
    mean, std = norm.fit(noise[5].flatten())
    print(mean, std**2)
    plt.savefig('noise_distribution_2.png')

    # plot the clean and noise pictures
    picture_clean = pictures_clean[5]
    picture_noise = pictures_noise[5]
    picture_raw_noise = np.abs(picture_noise.astype('int8') - picture_clean.astype('int8'))
    plt.subplot(3, 1, 1)
    plt.imshow(picture_clean)
    plt.subplot(3, 1, 2)
    plt.imshow(picture_noise)
    plt.subplot(3, 1, 3)
    plt.imshow(picture_raw_noise.astype('uint8'))
    plt.savefig('clean_noise_2.png')
    
    picture_noise_box = picture_noise.copy()
    picture_noise_box = cv2.blur(picture_noise, (11, 11))

    # plot the denosied pictures and the noise
    picture_noise_box_noise = picture_noise_box.astype('int8')  - picture_clean.astype('int8') 
    print(np.mean(np.abs(picture_noise_box.astype('int8')  - picture_clean.astype('int8'))))
    plt.subplot(4, 1, 1)
    plt.imshow(picture_noise)
    plt.subplot(4, 1, 2)
    plt.imshow(picture_noise_box)
    plt.subplot(4, 1, 3)
    plt.imshow(picture_clean)
    plt.subplot(4, 1, 4)
    plt.imshow(picture_noise_box_noise)
    plt.savefig('gaussian_noise_2.png')

    plt.figure()
    plt.hist(picture_noise_box_noise.flatten(), bins=100)
    plt.savefig('gaussian.png')