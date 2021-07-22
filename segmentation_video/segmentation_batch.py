import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.defchararray import split
from scipy import misc
import scipy
import skimage

def RGB2CYKB_v1(imgpath,markpath):
    cmyk_scale = 255
    img=Image.open(imgpath)
    r,g,b=img.split()
    r=np.array(r)
    g=np.array(g)
    b=np.array(b)
    c=1-r/255.
    m = 1 - g / 255.
    y = 1 - b / 255.
    x,yy=c.shape
    print(c.shape)
    min_cmy=np.zeros(c.shape)
    #print(min_cmy.shape,y[0,0])
    for i in range(x):
        for j in range(yy):
            #print(c[i,j], m[i,j], y[i,j])
            min_cmy[i,j] = min([c[i,j], m[i,j], y[i,j]])
    c = (c- min_cmy) / (1 - min_cmy)
    c=Image.fromarray(c*cmyk_scale).convert('RGB')
    c.save(markpath)

def thresh_otsu(imgpath,savepath):
    img=Image.open(imgpath)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret1,th1=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    cv2.imwrite(savepath,th1)
def thresh_BINARY_otsu(imgpath,savepath):
    img = cv2.imread(imgpath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret1,th1=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite(savepath,th1)
def wight_balance_v1(imgpath,savepath):
    # img=cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2HSV)
    # h,s,v=cv2.split(img)
    img=cv2.imread(imgpath)
    b,g,r=cv2.split(img)
    save=os.path.dirname(os.path.dirname(savepath))
    cv2.imwrite(os.path.join(save,'Isup_b.jpg'),b)
    cv2.imwrite(os.path.join(save,'Isup_g.jpg'),g)
    cv2.imwrite(os.path.join(save,'Isup_r.jpg'),r)
    #print(np.mean(s),np.mean(s[s>0]),np.sum(s>0),np.sum(s>=0))
    
    img=Image.open(imgpath)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    src_img = img
    b_gray, g_gray, r_gray = cv2.split(src_img)
    #I=0.299*r_gray+0.587*g_gray+0.114*b_gray
    # c='L'
    # if c=='L':
    #     #I=0.299*r_gray+0.587*g_gray+0.114*b_gray
    #     kr=0.96#np.mean(I)/np.mean(r_gray)
    #     kg=1.02#np.mean(I)/np.mean(g_gray)
    #     kb=1.02#np.mean(I)/np.mean(b_gray)
    # else:
    #     I=0.299*r_gray+0.587*g_gray+0.114*b_gray
    #     kr=np.mean(I)/np.mean(r_gray)
    #     kg=np.mean(I)/np.mean(g_gray)
    #     kb=np.mean(I)/np.mean(b_gray)

    I=0.299*r_gray+0.587*g_gray+0.114*b_gray
    kr=np.mean(I)/np.mean(r_gray)
    kg=np.mean(I)/np.mean(g_gray)
    kb=np.mean(I)/np.mean(b_gray)
    if kr>=0.96:kr=0.96
    if kg<=1.02:kg=1.02
    if kb<=1.02:kb=1.02

        
    print(kr,kg,kb)
    #np.mean(r_gray),np.mean(g_gray),np.mean(b_gray),
    result = cv2.merge([b_gray*kb, g_gray*kg, r_gray*kr])
    cv2.imwrite(savepath,result)

def gray_world(imgpath,markpath):
    img=Image.open(imgpath)
    x,y=img.size
    n=x*y
    #print(img.size)
    r,g,b=img.split()
    r=np.array(r)
    g=np.array(g)
    b=np.array(b)
    rmean=np.sum(r)/n
    gmean=np.sum(g)/n
    bmean=np.sum(b)/n
    #print(rmean,gmean,bmean)
    avg=np.mean([rmean,gmean,bmean])
    kr=avg/rmean
    kg=avg/gmean
    kb=avg/bmean
    r=np.expand_dims(r*kr,axis=2)
    g=np.expand_dims(g*kg,axis=2)
    b=np.expand_dims(b*kb,axis=2)
    img=np.concatenate((r,g,b),-1)
    #print(img.shape)
    img=Image.fromarray(np.uint8(img)).convert('RGB')
    img.save(markpath)

def red_substract_green(imgpath,markpath):
    img=Image.open(imgpath)
    r,g,b=img.split()
    #gg=np.array(g)[0,:50]
    #print(np.array(g)[0,:50],np.array(r)[0,:50],np.array(g)[0,:50]-np.array(r)[0,:50])
    # import sys
    # sys.exit(0)
    '''fun1'''
    # if white balance of L is same as H M, then using followed code
    # rsg=np.array(r).astype(np.float32)-np.array(g).astype(np.float32)
    # m_rsg=np.min(rsg)
    # h_rsg=rsg+abs(m_rsg)
    # rsg=(h_rsg).astype(np.uint8)
    # if c=='L':
    #     rsg[rsg>85]=255
    '''fun2'''
    rsg=np.array(g)-np.array(r)
    
    # plt.hist(rsg)#,bins #bins=np.arange(0,260,10)
    # plt.savefig('/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/L/hist_L_DAB.jpg')
    # plt.cla()
    
    # plt.hist(rsg)#,bins
    # plt.savefig('/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/L/hist_L_DAB85.jpg')
    #print(sorted(np.resize(rsg,(1024*1024))))
    # import sys
    # sys.exit(0)
    rsg=Image.fromarray(rsg).convert('RGB')
    rsg.save(markpath)

def closing_algorithm(imgpath,markpath):
    # img=Image.open(imgpath)
    # img=np.asarray(img,dtype=np.uint8)
    # img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img=cv2.imread(imgpath)
    kernal=np.ones((5,5),np.uint8)#(5,5)
    img_close=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernal)
    cv2.imwrite(markpath,img_close)

def medainblur(imgpath,markpath):
    img=Image.open(imgpath)
    img=cv2.cvtColor(np.asarray(img,dtype=np.uint8),cv2.COLOR_RGB2BGR)# 需要确认输入图片的mode
    img_medain=cv2.medianBlur(img,5)#5
    cv2.imwrite(markpath,img_medain)
def skeletin(imgpath,markpath):
    from skimage import morphology
    img=Image.open(imgpath)
    img=np.asarray(img)/255.
    img[img>0.5]=1
    img[img<0.5]=0

    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_skl=morphology.skeletonize(img)#input is bivalue 
    img_skl=np.array(Image.fromarray(img_skl).convert('L'))
    img_skl[img_skl>=100]=255
    img_skl[img_skl<100]=0
    Image.fromarray(img_skl).save(markpath)#.convert('L')

def fillHole(imgpath,markpath):
    img=np.array(Image.open(imgpath))
    img[img>100]=255
    img[img<100]=0
    img_copy=img.copy()
    h,w=img.shape
    mask=np.zeros((h+2,w+2),np.uint8)
    isbreak = False
    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            if(img_copy[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break
    cv2.floodFill(img_copy,mask,seedPoint,255)#img needs ing only 0,255 
    img_invert=cv2.bitwise_not(img_copy)
    img_out=img|img_invert
    Image.fromarray(img_out).save(markpath)
def opening_algorithm(imgpath,markpath):
    img=Image.open(imgpath)
    img=np.asarray(img,dtype=np.uint8)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    kernal=np.ones((5,5),np.uint8)
    img_close=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernal)
    cv2.imwrite(markpath,img_close)
def logical_AND(sklimgpath,open_fillhole_imgpath,markpath):
    skimg=cv2.cvtColor(cv2.imread(sklimgpath),cv2.COLOR_BGR2GRAY)
    skimg[skimg>=100]=255
    skimg[skimg<100]=0
    ofimg=cv2.cvtColor(cv2.imread(open_fillhole_imgpath),cv2.COLOR_BGR2GRAY)#1024,1024,3
    ofimg[ofimg<100]=0
    ofimg[ofimg>=100]=1
    outimg=skimg*ofimg
    cv2.imwrite(markpath,outimg)

def subtract(imgpath1,imgpath2,markpath):
    fir_img=np.array(Image.open(imgpath1))#很奇怪，即使保存前是二值的结果依然非二值，
    fir_img[fir_img>=100]=255#必须进行下面两行
    fir_img[fir_img<100]=0
    sec_img=np.array(Image.open(imgpath2))
    sec_img[sec_img>=100]=255
    sec_img[sec_img<100]=0
    out_img=fir_img-sec_img
    Image.fromarray(out_img).save(markpath)

def pseudo_color(imgpath,CMpath,BMpath,savepath):
    img=np.array(Image.open(imgpath))
    im=cv2.cvtColor(cv2.imread(CMpath),cv2.COLOR_BGR2GRAY)
    cmimg=cv2.cvtColor(np.array(Image.open(CMpath).convert('RGB')),cv2.COLOR_RGB2HSV)
    bmimg=cv2.cvtColor(np.array(Image.open(BMpath).convert('RGB')),cv2.COLOR_RGB2HSV)
    hsv_lower=(0,0,46)
    hsv_upper=(180,43,255)
    cmark=cv2.inRange(cmimg,hsv_lower,hsv_upper)
    bmark=cv2.inRange(bmimg,hsv_lower,hsv_upper)
    mask_c=cmark>0
    mask_b=bmark>0
    #cv2.cvtColor(cmark,cv2.COLOR_BGR2GRAY)
    img[mask_c]=(255,0,0)
    img[mask_b]=(0,255,0)
    Image.fromarray(img).save(savepath)

def CM_BM_ratio(imgCM,imgBM):
    fir_img=np.array(Image.open(imgCM))#很奇怪，即使保存前是二值的结果依然非二值，
    fir_img[fir_img<=100]=0#必须进行下面两行
    fir_img[fir_img>100]=1
    sec_img=np.array(Image.open(imgBM))
    sec_img[sec_img<=100]=0
    sec_img[sec_img>100]=1
    print(np.sum(fir_img),np.sum(sec_img))
    cm_bm_ratio=np.sum(fir_img)/np.sum(sec_img)
    return cm_bm_ratio

def blue_saturation(imgpath,savepath,savepath1):
    img=cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(img)
    wh=img.shape
    mark=np.zeros(wh)
    hsv_lower=(78,30,46)#100  78
    hsv_upper=(155,255,230)#124   155
    mark[cv2.inRange(img,hsv_lower,hsv_upper)>0]=255
    blue_num=np.sum(mark)
    cv2.imwrite(savepath,mark)
    # savepathh='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/BW_h.jpg'
    # savepaths='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/BW_s.jpg'
    # savepathv='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result/H/BW_v.jpg'
    # cv2.imwrite(savepathh,h)
    # cv2.imwrite(savepaths,s)
    # cv2.imwrite(savepathv,v)
    # import pdb
    # pdb.set_trace()
    # v=cv2.cvtColor(cv2.cvtColor(v,cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)
    # print(v[-1,:])
    #'/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/I_imgh.jpg'
    blue_ratio=blue_num/(wh[0]*wh[1])
    return blue_ratio

def haematoxylin_DAB_ratio(Negimg,Dabimg):
    negimg=cv2.imread(Negimg)
    dabimg=cv2.imread(Dabimg)
    negimg[negimg<=100]=0#必须进行下面两行
    negimg[negimg>100]=1
    dabimg[dabimg<=100]=0
    dabimg[dabimg>100]=1
    neg_dab_ratio=np.sum(negimg)/np.sum(dabimg)
    return neg_dab_ratio

def rgb_membrane(imgMem,Img):
    imgmem=cv2.cvtColor(cv2.imread(imgMem),cv2.COLOR_BGR2HSV)
    img=cv2.imread(Img)
    b,g,r=cv2.split(img)
    hsv_lower=(0,0,46)
    hsv_upper=(180,43,255)
    mark=cv2.inRange(imgmem,hsv_lower,hsv_upper)

    mean_b=np.mean(b[mark>0])
    mean_g=np.mean(g[mark>0])
    mean_r=np.mean(r[mark>0])
    return mean_b,mean_g,mean_r

    # kernel=np.ones((5,5),np.uint8)
    # erosion_mem=cv2.erode(imgmem,kernel,iterations=2)

    # hsv_lower=(0,0,46)
    # hsv_upper=(180,43,255)
    # mark=cv2.inRange(erosion_mem,hsv_lower,hsv_upper)
    # img[mark>0]=(0,0,255)
    # cv2.imwrite('/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/I_membrane_mark_erode2.jpg',img)
def get_cell_inhole(cellpath,holepath,savepath):
    cell=cv2.cvtColor(cv2.imread(cellpath),cv2.COLOR_BGR2GRAY)
    hole=cv2.cvtColor(cv2.imread(holepath),cv2.COLOR_BGR2GRAY)
    hole[hole<=100]=0
    hole[hole>=100]=1
    cell*=hole
    cv2.imwrite(savepath,cell)
def get_marks_noclass(imgpath,saveroot):
    Bsavepath=os.path.join(saveroot,'I3_wight_balance.jpg')
    wight_balance_v1(imgpath,Bsavepath)
    # Gmarkpath=os.path.join(saveroot,'I3_gray_wight_balance.jpg')
    # gray_world(Bsavepath,Gmarkpath)
    DABmarkpath=os.path.join(saveroot,'I4_DAB_gr.jpg')
    red_substract_green(Bsavepath,DABmarkpath)
    Memarkpath=os.path.join(saveroot,'I5_Mem.jpg')
    thresh_otsu(DABmarkpath,Memarkpath)
    Memclospath=os.path.join(saveroot,'I6_Memclos.jpg')
    closing_algorithm(Memarkpath,Memclospath)
    MemclosMedainpath=os.path.join(saveroot,'I6_MemclosMedain.jpg')
    medainblur(Memclospath,MemclosMedainpath)
    skelpath=os.path.join(saveroot,'I7_skeletin.jpg')
    skeletin(MemclosMedainpath,skelpath)
    FillHolepath=os.path.join(saveroot,'I8_fillhole.jpg')
    fillHole(skelpath,FillHolepath)
    fhOpenpath=os.path.join(saveroot,'I8_fillholeopen.jpg')
    opening_algorithm(FillHolepath,fhOpenpath)
    CMpath=os.path.join(saveroot,'I8_CM.jpg')
    logical_AND(skelpath,fhOpenpath,CMpath)
    BMpath=os.path.join(saveroot,'I9_BM.jpg')
    subtract(skelpath,CMpath,BMpath)
    pseupath=os.path.join(saveroot,'I10_pseudo.jpg')
    pseudo_color(imgpath,CMpath,BMpath,pseupath)
def get_marks(imgpath,saveroot):
    '''step 1-2'''
    Cmarkpath=os.path.join(saveroot,'I1_C_cymk.jpg')
    RGB2CYKB_v1(imgpath,Cmarkpath)

    Nsavepath=os.path.join(saveroot,'I2_negative_cell.jpg')
    thresh_otsu(Cmarkpath,Nsavepath)
    '''step 3-10'''
    Bsavepath=os.path.join(saveroot,'I3_wight_balance.jpg')
    wight_balance_v1(imgpath,Bsavepath)
    # Gmarkpath=os.path.join(saveroot,'I3_gray_wight_balance.jpg')
    # gray_world(Bsavepath,Gmarkpath)
    DABmarkpath=os.path.join(saveroot,'I4_DAB_gr.jpg')
    red_substract_green(Bsavepath,DABmarkpath)
    Memarkpath=os.path.join(saveroot,'I5_Mem.jpg')
    thresh_otsu(DABmarkpath,Memarkpath)
    Memclospath=os.path.join(saveroot,'I6_Memclos.jpg')
    closing_algorithm(Memarkpath,Memclospath)
    MemclosMedainpath=os.path.join(saveroot,'I6_MemclosMedain.jpg')
    medainblur(Memclospath,MemclosMedainpath)
    skelpath=os.path.join(saveroot,'I7_skeletin.jpg')
    skeletin(MemclosMedainpath,skelpath)
    FillHolepath=os.path.join(saveroot,'I8_fillhole.jpg')
    fillHole(skelpath,FillHolepath)
    fhOpenpath=os.path.join(saveroot,'I8_fillholeopen.jpg')
    opening_algorithm(FillHolepath,fhOpenpath)
    CMpath=os.path.join(saveroot,'I8_CM.jpg')
    logical_AND(skelpath,fhOpenpath,CMpath)
    BMpath=os.path.join(saveroot,'I9_BM.jpg')
    subtract(skelpath,CMpath,BMpath)
    pseupath=os.path.join(saveroot,'I10_pseudo.jpg')
    pseudo_color(imgpath,CMpath,BMpath,pseupath)
def get_cell(imgpath,savepath):
    Bsavepath=os.path.join(savepath,'I3_wight_balance.jpg')
    wight_balance_v1(imgpath,Bsavepath)
    DABmarkpath=os.path.join(savepath,'I4_DAB_gr_.jpg')
    red_substract_green(Bsavepath,DABmarkpath)
    FHpath=os.path.join(savepath,'I8_fillholeopen.jpg')
    #BMpath=os.path.join(savepath,'I9_BM.jpg')
    #Memarkpath=os.path.join(savepath,'I5_Mem.jpg')
    bspath=os.path.join(savepath,'I_blue_cyan_purple_S30V230_BW.jpg')
    Hpath=os.path.join(savepath,'I_H.jpg')
    # markpath=os.path.join(savepath,'I_H_thre_Bin_otsu.jpg')
    # thresh_BINARY_otsu(Hpath,markpath)
    imgpath=os.path.join(savepath,'I3_wight_balance.jpg')
    blue_saturation(imgpath,bspath,Hpath)
    #bspath=os.path.join(savepath,'I_blue_cyan_purple_S10.jpg')
    BHpath=os.path.join(savepath,'I_blueinhole.jpg')
    #get_cell_inhole(bspath,FHpath,BHpath)
    # Memclospath=os.path.join(savepath,'I_blueCP_S30V230_close9.jpg')
    # closing_algorithm(bspath,Memclospath)
    # markpath=os.path.join(savepath,'I_blueCP_S30V230_close9_medain5.jpg')
    # medainblur(Memclospath,markpath)
    markpath=os.path.join(savepath,'I_blueCP_S30V230_BW_medain5.jpg')
    medainblur(bspath,markpath)
    Memclospath=os.path.join(savepath,'I_blueCP_S30V230_BW_medain5_close5.jpg')
    closing_algorithm(markpath,Memclospath)
    
def get_feature(imgpath,CMpath,BMpath,Memarkpath,bspath,Hpath):
    saveroot='/'.join(imgpath.split('/')[:-2])
    CB_ratio=CM_BM_ratio(CMpath,BMpath)

    blue_ares=blue_saturation(imgpath,bspath,Hpath)
    #print(blue_ares,bspath)
    Negimg=bspath
    neg_dab_ratio=haematoxylin_DAB_ratio(Negimg,Memarkpath)
    #print(neg_dab_ratio)
    mean_b,mean_g,mean_r=rgb_membrane(Memarkpath,imgpath)
    #print(mean_b,mean_g,mean_r)
    return [CB_ratio,blue_ares,mean_b,mean_g,mean_r]
def get_intensity(DABpath,imgpath):
    dabimg=cv2.cvtColor(cv2.imread(DABpath),cv2.COLOR_BGR2GRAY)
    dabimg[dabimg<=100]=0
    dabimg[dabimg>100]=1
    img=cv2.imread(imgpath)
    r,g,b=cv2.split(img)
    intensity=np.mean(r*dabimg)
    print(intensity)


if __name__=='__main__':
    '''get marks'''
    # dataroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/data'
    # saveroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result'
    # for c in os.listdir(dataroot):
    #     print(c)
    #     # if c!='L':
    #     #     continue
    #     for f in os.listdir(os.path.join(dataroot,c)):
    #         imgpath=os.path.join(dataroot,c,f)
    #         savepath=os.path.join(saveroot,c,f)
    #         if not os.path.exists(savepath):
    #             os.makedirs(savepath)
    #         get_marks(imgpath,savepath)
    '''get cell'''
    # dataroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/data'
    # saveroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result'
    # for c in os.listdir(dataroot):
    #     print(c)
    #     # if c!='H':
    #     #     continue
    #     for f in os.listdir(os.path.join(dataroot,c)):
    #         imgpath=os.path.join(dataroot,c,f)
    #         savepath=os.path.join(saveroot,c,f)
    #         if not os.path.exists(savepath):
    #             os.makedirs(savepath)
    #         #get_marks(imgpath,savepath)

    #         get_cell(imgpath,savepath)
    '''get intensity'''
    dataroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/data'
    saveroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result'
    for c in os.listdir(dataroot):
        print(c)
        # if c!='H':
        #     continue
        for f in os.listdir(os.path.join(dataroot,c)):
            imgpath=os.path.join(dataroot,c,f)
            savepath=os.path.join(saveroot,c,f)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            #get_marks(imgpath,savepath)
            DABpath=os.path.join(savepath,'I4_DAB_gr.jpg')
            get_intensity(DABpath,imgpath)
            #get_feature(imgpath,CMpath,BMpath,Memarkpath,bspath,Hpath)
    '''get marks using no class data'''
    # dataroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/data_more'
    # saveroot='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/membrane/try_train/result_more/fix_WB/'
    # for f in os.listdir(dataroot):
    #     print(f)
    #     imgpath=os.path.join(dataroot,f)
    #     savepath=os.path.join(saveroot,f)
    #     if not os.path.exists(savepath):
    #         os.makedirs(savepath)
    #     get_marks_noclass(imgpath,savepath)




