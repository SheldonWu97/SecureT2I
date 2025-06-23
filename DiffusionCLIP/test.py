import torch
import torch.nn as nn

def func1():
    fid = [254.7 ,  235 ,232.5 ,  224.6  , 222.3  , 216.5 ,  214.9  , 212 ,209.2,265.4 ,  235.9  , 234.6  , 230.7 ,  222.2  , 221.6   ,264.9 ,  285.1,   342.2]
    i_s = [1 ,  1.44  ,  1.62  ,  1.79   , 1.9 ,1.95  ,  1.97  ,  2.03 ,   2.05,1  , 1.51  ,  1.76  ,  1.95  ,  1.54  ,  2.2 ,2.34   , 3.15 ,   3.16]
    clip = [0.522 ,  0.518 ,  0.52 ,   0.524 ,  0.524  , 0.526  , 0.532  , 0.532 ,  0.53,0.504 ,  0.51  ,  0.512 ,  0.52  ,  0.52  ,  0.512 ,  0.5 ,0.504   ,0.464]


    data_fid = torch.Tensor(fid)
    data_is = torch.Tensor(i_s)
    data_clip = torch.Tensor(clip)

    norm_fid = (data_fid.min() - data_fid) / (data_fid.max() - data_fid.min())
    norm_is = (data_is - data_is.min()) / (data_is.max() - data_is.min())
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