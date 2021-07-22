from numpy.lib.npyio import save
import scipy.ndimage as ndimg
import numpy as np
from numba import jit
import cv2
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

@jit  # trans index to r, c...

def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    return rst


#@jit  # fill a node (may be two or more points)
def fill(img, msk, p, nbs, buf):
    msk[p] = 3
    buf[0] = p
    back = img[p]
    cur = 0
    s = 1
    while cur < s:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if not cp<1048576:
                continue
            if img[cp] == back and msk[cp] == 1:
                msk[cp] = 3
                buf[s] = cp
                s += 1
                if s == len(buf):
                    buf[:s - cur] = buf[cur:]
                    s -= cur
                    cur = 0
        cur += 1
    #msk[p] = 3


 #@jit  # my mark
 
def mark(img, msk, buf, mode):  # mark the array use (0, 1, 2)
    omark = msk     
    nbs = neighbors(img.shape)
    idx = np.zeros(1024 * 1024, dtype=np.int64)#128 to024 1
    img = img.ravel()  # 降维
    msk = msk.ravel()  # 降维
    s = 0
    for p in range(len(img)):
        if msk[p] != 1: continue  
        flag = False             
        for dp in nbs:
            if not (p + dp)<1048576:
                continue
            if mode and img[p + dp] > img[p]: 
                flag = True
                break
            elif not mode and img[p + dp] < img[p]:
                flag = True
                break
        
        if flag : continue
        else    : fill(img, msk, p, nbs, buf)
        idx[s] = p
        s += 1
        if s == len(idx): break
    plt.imshow(omark, cmap='gray')
    return idx[:s].copy()

def filter(img, msk, idx, bur, tor, mode):
    omark = msk  
    nbs = neighbors(img.shape)
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    msk = msk.ravel()

    arg = np.argsort(img[idx])[::-1 if mode else 1] 

    for i in arg:
        if msk[idx[i]] != 3:    
            idx[i] = 0
            continue
        cur = 0
        s = 1
        bur[0] = idx[i] 
        while cur < s:
            p = bur[cur]
            if msk[p] == 2:     
                idx[i] = 0
                break

            for dp in nbs:
                cp = p + dp
                if not (p + dp)<1048576:
                    continue
                if msk[cp] == 0 or cp == idx[i] or msk[cp] == 4: continue
                if mode and img[cp] < img[idx[i]] - tor: continue
                if not mode and img[cp] > img[idx[i]] + tor: continue
                bur[s] = cp
                s += 1
                if s == 1024 * 1024:
                    cut = cur // 2
                    msk[bur[:cut]] = 2
                    bur[:s - cut] = bur[cut:]
                    cur -= cut
                    s -= cut

                if msk[cp] != 2: msk[cp] = 4    
            cur += 1
        msk[bur[:s]] = 2    
        #plt.imshow(omark, cmap='gray')

    return idx2rc(idx[idx > 0], acc)

def find_maximum(img, tor, mode=True):
    msk = np.zeros_like(img, dtype=np.uint8)  
    msk[tuple([slice(1, -1)] * img.ndim)] = 1  
    buf = np.zeros(1024 * 1024, dtype=np.int64)#128 to 1024
    omark = msk
    idx = mark(img, msk, buf, mode)
    plt.imshow(msk, cmap='gray')
    idx = filter(img, msk, idx, buf, tor, mode)
    return idx

 
if __name__ == '__main__':
    #from scipy.misc import imread
    from scipy.ndimage import gaussian_filter
    from time import time
    import matplotlib.pyplot as plt
    import os
    dataroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/data'
    saveroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result'
    for c in os.listdir(dataroot):
        print(c)
        if c!='L':
            continue
        for f in os.listdir(os.path.join(dataroot,c)):
            imgpath=os.path.join(dataroot,c,f)
            savepath=os.path.join(saveroot,c,f)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            img = cv2.imread(os.path.join(savepath,'I_blueCP_S30V230_BW_medain5_close5.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img[img<=100]=0
            # img[img>100]=255
            ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img[:] = ndimg.distance_transform_edt(img)
            #plt.imshow(img, cmap='gray')
            #pts = find_maximum(img, 20, True)
            start = time()
            pts = find_maximum(img, 6, True)
            print(time() - start)

            #plt.imshow(img, cmap='gray')
            img_ori=cv2.imread(imgpath)
            img_ori=cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB)
            plt.imshow(img_ori)#, cmap='gray'
            plt.plot(pts[:, 1], pts[:, 0], 'r.',markersize=1)#, markersize=0.3
            
            plt.show()
            plt.savefig(os.path.join(saveroot,c,'localmax6_S30V230_BW_medain5_close5_marksize1r_imgori.jpg'))
            plt.cla()


    # img = cv2.imread('/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/img_690799-3 her2 - 2019-06-18 21.10.03_level_0_ul_x_52167_ul_y_8644.jpg/I_blueCP_S30V230_BW_medain5_close5.jpg')
    # #/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/img_690799-3 her2 - 2019-06-18 21.10.03_level_0_ul_x_52167_ul_y_8644.jpg/I_blue_cyan_purple_S30V230_BW.jpg
    # # /home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/img_690799-3 her2 - 2019-06-18 21.10.03_level_0_ul_x_52167_ul_y_8644.jpg/I_H.jpg
    # # /home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/M/img_745195-2 HER-2 - 2019-05-29 16.19.45_level_0_ul_x_81030_ul_y_36270.jpg/I_blue_cyan_purple_S10.jpg
    # #/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/L/img_694543-3 her-2 - 2019-06-17 15.31.23_level_0_ul_x_61360_ul_y_25011.jpg/I_blue_cyan_purple_S10.jpg
    # #/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/img_690799-3 her2 - 2019-06-18 21.10.03_level_0_ul_x_52167_ul_y_8644.jpg/I_blue_cyna_purpleS0.jpg
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img[img<=100]=0
    # # img[img>100]=255
    # ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # img[:] = ndimg.distance_transform_edt(img)
    # #plt.imshow(img, cmap='gray')
    # #pts = find_maximum(img, 20, True)
    # start = time()
    # pts = find_maximum(img, 6, True)
    # print(time() - start)

    # #plt.imshow(img, cmap='gray')
    # img_ori=cv2.imread('/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/data/H/img_690799-3 her2 - 2019-06-18 21.10.03_level_0_ul_x_52167_ul_y_8644.jpg')
    # img_ori=cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB)
    # plt.imshow(img_ori)#, cmap='gray'
    # plt.plot(pts[:, 1], pts[:, 0], 'r.',markersize=1)#, markersize=0.3
    # plt.show()
    # plt.savefig('/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/localmax6_S30V230_BW_medain5_close5_marksize1r_imgori.jpg')
