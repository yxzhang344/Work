import cv2
import os
# path='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/img_690799-3 her2 - 2019-06-18 21.10.03_level_0_ul_x_52167_ul_y_8644.jpg/I4_DAB_gr.jpg'
# savepath='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/img_690799-3 her2 - 2019-06-18 21.10.03_level_0_ul_x_52167_ul_y_8644.jpg/I4_DAB_gr2rgb.jpg'
# im=cv2.imread(path)
# im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
# cv2.imwrite(savepath,im)
import numpy as np
import json
def generate_json(fillhole_path,cm_path,bm_path,savepath):
    fillhole=cv2.cvtColor(cv2.imread(fillhole_path),cv2.COLOR_BGR2GRAY)
    cmimg=cv2.cvtColor(cv2.imread(cm_path),cv2.COLOR_BGR2GRAY)
    bmimg=cv2.cvtColor(cv2.imread(bm_path),cv2.COLOR_BGR2GRAY)
    filllist=np.where(fillhole>=100)
    cmlist=np.where(cmimg>=100)
    bmlist=np.where(bmimg>=100)
    fillxy=[]
    for i in range(len(filllist[0])):
        fillxy.append([str(filllist[0][i]),str(filllist[1][i])])
    cmxy=[]
    for i in range(len(cmlist[0])):
        cmxy.append([str(cmlist[0][i]),str(cmlist[1][i])])
    bmxy=[]
    for i in range(len(bmlist[0])):
        bmxy.append([str(bmlist[0][i]),str(bmlist[1][i])])
    dict_coord = {'completed_filled': fillxy,
                      'completed': cmxy,
                      'incompleted': bmxy
                     }
    with open(savepath,'w') as f:
        json.dump(dict_coord,f)



    

if __name__=='__main__':
    dataroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result_more/fix_WB'
    saveroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result_more/cuicui_json/'
    for f in os.listdir(dataroot):
        print(f)
        filpath=os.path.join(dataroot,f)
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)
        fillhole=os.path.join(filpath,'I8_fillholeopen.jpg')
        cm=os.path.join(filpath,'I8_CM.jpg')
        bm=os.path.join(filpath,'I9_BM.jpg')
        savepath=os.path.join(saveroot,f+'.json')
        generate_json(fillhole,cm,bm,savepath)
    


