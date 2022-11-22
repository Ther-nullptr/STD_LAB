from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
from tqdm import tqdm
import torch
import os

img_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Test/Clean/img'
img_names = os.listdir(img_dir)
img_names.sort()

for img_name in tqdm(img_names):
    print(img_name)
    image = Image.open(os.path.join(img_dir, img_name))
    feature_extractor = BeitFeatureExtractor.from_pretrained('/root/kyzhang/download/beit-base-patch16-224-pt22k-ft22k')
    print(feature_extractor)
    model = BeitForImageClassification.from_pretrained('/root/kyzhang/download/beit-base-patch16-224-pt22k-ft22k')
    print(model)
    inputs = feature_extractor(images=image, return_tensors="pt")
    print(inputs['pixel_values'].shape)
    x = torch.randn((10, 3, 224, 224))
    outputs = model.beit.forward(x)
    print(outputs.pooler_output.shape)
    # logits = outputs.logits
    # # model predicts one of the 21,841 ImageNet-22k classes
    # prob, predicted_class_idx = logits.topk(k=5, dim=-1)
    # for i in range(len(predicted_class_idx[0])):
    #     print("Predicted class:", model.config.id2label[int(predicted_class_idx[0][i])])
    #     print("Probility:", float(prob[0][i]))
