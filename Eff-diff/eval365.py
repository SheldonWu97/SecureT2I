import os
import glob
import argparse
#from cleanfid import fid
from base_dataset import BaseDataset
from metric import inception_score, inception_score_place365
from torchvision import transforms as trn
import random
import pdb
from pytorch_fid import fid_score

import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

def clip_dist(folder1, folder2):
    def get_clip_embeddings(image_paths):
        embeddings = []
        for image_path in tqdm(image_paths):
            image = Image.open(image_path)
            image = preprocess(image).unsqueeze(0)  # 添加batch维度
            with torch.no_grad():
                outputs = model.get_image_features(pixel_values=image)
            embeddings.append(outputs.squeeze(0))
        return torch.stack(embeddings)

    def get_image_paths(folder):
        return [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('png', 'jpg', 'jpeg'))]
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275)),
    ])

    image_paths1 = get_image_paths(folder1)
    image_paths2 = get_image_paths(folder2)
    embeddings1 = get_clip_embeddings(image_paths1)
    embeddings2 = get_clip_embeddings(image_paths2)

    cosine_similarities = F.cosine_similarity(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), dim=-1)
    average_cosine_similarity = cosine_similarities.mean().item()
    #print(f"\033[32mClip embedding distance: {average_cosine_similarity}\033[0m")
    return average_cosine_similarity

def prepare_blur(attr):
    input_folder = './out/' + attr + '/forget/orig'
    output_folder = './out/' + attr + '/forget/blur'

    if os.path.exists(output_folder):
        return 0
    else:
        os.makedirs(output_folder)

    transform = transforms.Compose([
        transforms.Resize((16,16)),
        transforms.Resize((256,256))])

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = Image.open(input_path)
            blurred_image = transform(image)
            blurred_image.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dst', type=str, default='./unl_out/forget', help='Generated images directory')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='masking ratio w.r.t. one dimension')
    args = parser.parse_args()

    tfs = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    attr = 'bedroom_princess_'
    prepare_blur(attr)

    #src_list = ['/retain/orig', '/retain/orig', '/forget/orig', '/forget/orig']
    #dst_list = ['/retain/pre', '/retain/unl', '/forget/pre', '/forget/unl']
    #src_list = ['/retain/orig', '/forget/orig']
    #dst_list = ['/retain/unl', '/forget/unl']
    src_list = ['/forget/blur']
    dst_list = ['/retrain_forget']
    #dst_list = ['/forget/unl']

    fid_list, is_list, clip_list = [], [], []
    for i in range(len(dst_list)):
        dst = './out/' + attr + dst_list[i]
        src = './out/' + attr + src_list[i]

        score = fid_score.calculate_fid_given_paths([src, dst], batch_size=50, device='cuda', dims=2048)
        try:
            is_mean, is_std = inception_score_place365(BaseDataset(dst, tfs=tfs), cuda=True, batch_size=8, resize=False, splits=10)
        except:
            is_mean, is_std = inception_score(BaseDataset(dst, tfs=tfs), cuda=True, batch_size=8, resize=False, splits=10)
        #metric=[fid_score, is_mean, is_std]
        print(src, dst)
        #print("\033[32mFID: {}\033[0m".format(fid_score))
        #print('\033[32mIS:{} {}\033[0m'.format(is_mean, is_std))
        fid_list.append(score)
        is_list.append([is_mean, is_std])

        clip = clip_dist(src, dst)
        clip_list.append(clip)

    print('FID: ', fid_list)
    print('IS: ', is_list)
    print('Clip: ', clip_list)
