import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def forget_rate():
    forget_rate = ['1:9','2:8','3:7','4:6','5:5','6:4','7:3','8:2','9:1']
    fid_orig_re = [210.038,214.172,215.756,219.478,194.05,228.808,232.558,243.324,264.576]
    fid_re = [205.49,203.174,205.298,213.958,210.71,222.71,272.876,302.418,370.336]
    fid_orig_fo = [254.67,234.95,232.45,224.588,222.3,216.52,214.894,211.972,209.216]
    fid_fo = [265.35,235.9,234.602,230.746,222,221.604,264.86,285.07,342.152]

    is_orig_re = [2.062,1.996,1.934,1.852,1.6,1.7,1.612,1.392,1]
    is_re = [2.108,2.146,1.99,2.018,1.48,1.828,1.616,1.512,1]
    is_orig_fo = [1,1.436,1.622,1.792,1.9,1.95,1.97,2.026,2.048]
    is_fo = [1,1.508,1.764,1.946,1.54,2.202,2.344,3.148,3.162]

    clip_orig_re = [0.532,0.534,0.534,0.534,0.54,0.54,0.534,0.524,0.544]
    clip_re = [0.526,0.53,0.53,0.53,0.53,0.526,0.5,0.488,0.454]
    clip_orig_fo = [0.522,0.518,0.52,0.524,0.52,0.526,0.532,0.532,0.53]
    clip_fo = [0.504,0.51,0.512,0.52,0.52,0.512,0.5,0.504,0.464]
    
        # 创建一个图形
    plt.figure(figsize=(10, 6))

    # 绘制三条折线
    plt.plot(forget_rate, fid_orig_re, marker='o', linestyle='-', linewidth=3.5, color='#FFCDD2', label='Origin Retain')
    plt.plot(forget_rate, fid_re, marker='o', linestyle='-', linewidth=3.5, color='#D32F2F', label='Retain')
    plt.plot(forget_rate, fid_orig_fo, marker='o', linestyle='-', linewidth=3.5, color='#BBDEFB', label='Origin Forget')
    plt.plot(forget_rate, fid_fo, marker='o', linestyle='-', linewidth=3.5, color='#1976D2', label='Forget')

    # 设置标题和标签
    plt.title('FID for different forget-retain rate', fontsize=50)
    plt.xlabel('Forget-Retain Rate', fontsize=50)
    plt.ylabel('FID', fontsize=50)

    # 设置坐标轴刻度
    plt.xticks(forget_rate, fontsize=50)
    plt.yticks(fontsize=50)

    # 显示网格和图例
    plt.legend(fontsize=50)

    # 显示图形
    plt.show()

if __name__ == '__main__':
    forget_rate()