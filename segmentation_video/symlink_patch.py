import os
import random
def symlink(patchpath,symlinkpath):
    patches=os.listdir(patchpath)
    wsi=[]
    for p in patches:
        wsi_name=p.split('-')[0]
        if not wsi_name in wsi:
            wsi.append(wsi_name)

    w_patch={}
    for p in patches:
        wsi_name=p.split('-')[0]
        if wsi_name in w_patch:
            w_patch[wsi_name].append(p)
        else:
            w_patch[wsi_name]=[p]
    if not os.path.exists(symlinkpath):
        os.makedirs(symlinkpath)
    for w in w_patch:
        n_patch=len(w_patch[w])
        index=random.randint(0,n_patch-1)
        print(index,n_patch)
        select_patch=w_patch[w][index]
        os.symlink(os.path.join(patchpath,select_patch),os.path.join(symlinkpath,select_patch))
    print(len(wsi))
if __name__=='__main__':
    patchpath='/mnt/share/quantitative_analysis_database/VOC_her2/VOC2007_1024_real_nucleus/JPEGImages'
    symlinkpath='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/data_more'
    symlink(patchpath,symlinkpath)
