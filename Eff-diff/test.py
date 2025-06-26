import torch
import torch.nn as nn

def func1():
    fid = [462.334,553.402,514.372,518.754,482]
    i_s = [3.036,2.756,3.542,3.376,3.62]
    clip = [0.548,0.756,0.694,0.692,0.646]

    data_fid = torch.Tensor(fid)
    data_is = torch.Tensor(i_s)
    data_clip = torch.Tensor(clip)

    norm_fid = (data_fid.min() - data_fid) / (data_fid.max() - data_fid.min())
    norm_is = (-data_is + data_is.min()) / (data_is.max() - data_is.min())
    norm_clip = (data_clip - data_clip.min()) / (data_clip.max() - data_clip.min())

    score = (norm_fid + norm_is + norm_clip) / 3.0
    print(score)

def func2():
    orig = torch.Tensor([0.537 ,  0.495])
    cand = torch.Tensor([[-0.333 , -0.333],[0.117  , 0.130],[0.042 ,  0.098],[0.434 ,  0.546]]) #nx2

    score = torch.zeros(cand.size(0))
    for i, can in enumerate(cand):
        score[i] = (1-orig[0]+can[0])/torch.max(orig[0],can[0]) + \
                   (orig[1]-can[1]) / torch.max(orig[1],can[1])
    print(score)

if __name__ == '__main__':
    func1()