import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PIL import Image
import pdb
from transformers import CLIPProcessor, CLIPModel
from sklearn.manifold import TSNE
from torchvision import transforms

def clip(folder):
    def get_clip_embeddings(image_paths):
        embeddings = []
        for image_path in tqdm(image_paths):
            image = Image.open(image_path)
            image = preprocess(image).unsqueeze(0)  # 添加batch维度
            with torch.no_grad():
                outputs = model.get_image_features(pixel_values=image)
            embeddings.append(outputs.squeeze(0))
        return torch.stack(embeddings).numpy()
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

    image_path = get_image_paths(folder)
    embeddings = get_clip_embeddings(image_path)
    return embeddings

if __name__ == "__main__":
    edit_attr = 'beards_retain'
    unlearn_forget_norm = clip('./out/'+edit_attr+'/forget/unl')+0.2
    unlearn_retain_norm = clip('./out/'+edit_attr+'/retain/unl')
    original_forget_norm = clip('./out/'+edit_attr+'/forget/pre')+0.2
    original_retain_norm = clip('./out/'+edit_attr+'/retain/pre')

    unlearn_forget_label, unlearn_retain_label = np.zeros(unlearn_forget_norm.shape[0])+0.0, np.zeros(unlearn_retain_norm.shape[0])+1.0
    original_forget_label, original_retain_label = np.zeros(original_forget_norm.shape[0])+2.0, np.zeros(original_retain_norm.shape[0])+3.0

    X = np.concatenate((unlearn_forget_norm, unlearn_retain_norm, original_forget_norm, original_retain_norm), axis=0)
    
    del unlearn_forget_norm
    del unlearn_retain_norm
    del original_forget_norm
    del original_retain_norm

    labels = np.concatenate((unlearn_forget_label, unlearn_retain_label, original_forget_label, original_retain_label), axis=0).tolist()
    tsne = TSNE(n_components=2, random_state=0)

    with torch.no_grad():
        Y = tsne.fit_transform(X)

    #plt.rcParams['font.size']=20
    plt.rcParams['figure.figsize']=(16,12)

    fig, ax = plt.subplots()
    mi=['s','s','x','x']
    ci=['red','blue','green','#ED7D31']
    size=[40,40,40,40]
    linewidths=[1,1,4,4]
    labels = ['Disable-Forbid Set', 'Disable-Permit Set', 'Original-Forbid Set', 'Original-Permit Set']
    # labels = ['', '', '', '']

    for i, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:pink']):
        ax.scatter(Y[i*len(unlearn_forget_label):(i+1)*len(unlearn_forget_label), 0], Y[i*len(unlearn_forget_label):(i+1)*len(unlearn_forget_label), 1], \
                s=size[i]*4.0, marker=mi[i], color=ci[i], label=labels[i], edgecolors='none', alpha=1.0, linewidths=linewidths[i])
    ax.legend(fontsize=40)
    ax.grid(True)
    
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    #plt.title('Retain Label', fontsize=50)
    plt.tight_layout()

    plt.savefig('new_retain.png')
    plt.show()