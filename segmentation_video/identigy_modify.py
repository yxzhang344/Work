import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal

def get_video(rootpath):
    '''逐帧使用颜色阈值得到cell像素，使用轮廓得到bbox并保存回video'''
    font = cv2.FONT_HERSHEY_SIMPLEX
    lower_white = np.array([0, 0, 100]) # 绿色范围低阈值
    upper_white = np.array([180, 30, 255]) # 绿色范围高阈值

    #需要更多颜色，可以去百度一下HSV阈值！
    cap = cv2.VideoCapture(os.path.join(rootpath,"corrected_001.avi"))#打开USB摄像头
    fps=7
    size=(512,512)
    videoWriter =cv2.VideoWriter(os.path.join(rootpath,'corrected_001_result_V100.avi'),cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)
    num = 0

    while (True):
        ret, frame = cap.read() # 读取一帧
        #cv2.imwrite(os.path.join(rootpath,'Frame',"%d.jpg"%num), frame)
        if ret == False: # 读取帧失败
            break
        num = num + 1
        # if num!=25:
        #     continue
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv_img)
        #only v has result
        # cv2.imwrite(os.path.join(rootpath,'25_v.jpg'),v)
        v_median = cv2.medianBlur(v, 7)
        #cv2.imwrite(os.path.join(rootpath,'25_v_median.jpg'),v_median)
        mask_white = cv2.inRange(hsv_img, lower_white, upper_white) # 根据颜色范围删选
        # 根据颜色范围删选
        #cv2.imwrite(os.path.join(rootpath,'25_mask_white.jpg'),mask_white)
        mask_white = cv2.medianBlur(mask_white, 7) # 中值滤波

        mask_white, contours, hierarchy = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #cv2.imwrite(os.path.join(rootpath,'25_mask_white_mediam.jpg'),mask_white)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            #cv2.putText(frame, "cell", (x, y - 5), font, 0.7, (0, 255, 0), 2)
        
        videoWriter.write(frame)
        # cv2.imshow("dection", frame)
        cv2.imwrite(os.path.join(rootpath,'Frame_result',"%d.jpg"%num), frame)
        
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

def use_threshold_getcell(imgpath,savepath):
    img=cv2.imread(imgpath)
    lower_white = np.array([0, 0, 100]) # 绿色范围低阈值
    upper_white = np.array([180, 30, 255]) # 绿色范围高阈值
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #cv2.imwrite(os.path.join(rootpath,'25_v_median.jpg'),v_median)
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white) # 根据颜色范围删选
    # 根据颜色范围删选
    #cv2.imwrite(os.path.join(rootpath,'25_mask_white.jpg'),mask_white)
    mask_white = cv2.medianBlur(mask_white, 7) # 中值滤波

    mask_white, contours, hierarchy = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.imwrite(os.path.join(rootpath,'25_mask_white_mediam.jpg'),mask_white)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
    cv2.imwrite(savepath,img)
def get_all_frame_max_mask(rootpath):
    '''得到所有帧max像素生成的image'''
    cap = cv2.VideoCapture(os.path.join(rootpath,"corrected_001.avi"))
    frame_v=[]
    while (True):
        ret,frame=cap.read()
        if ret==False:
            break
        _,_,v=cv2.split(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV))
        frame_v.append(v)
        if cv2.waitKey(20) & 0xFF==27:
            break
    cap.release()
    max_v=np.array(frame_v).max(0)
    cv2.imwrite(os.path.join(rootpath,'max_frame.jpg'),max_v)
def fillHole(imgpath,markpath):
    img=cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2GRAY)
    img[img>100]=255
    img[img<100]=0
    img_copy=img.copy()
    hw=img.shape
    mask=np.zeros((hw[0]+2,hw[1]+2),np.uint8)
    isbreak = False
    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            print(img_copy[i][j])
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
def get_outline(markpath,savepath):
    img=cv2.cvtColor(cv2.imread(markpath),cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img,(3,3),0)
    cv2.imwrite(os.path.join(rootpath,'max_frame_gaussian.jpg'),img1)
    canny = cv2.Canny(img1, 150, 255)
    cv2.imwrite(savepath,canny)
    mask_white, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(mask_white, (x, y), (x + w, y + h), (0, 255, 255), 1)
    #cv2.imwrite(savepath,mask_white)
# 生成高斯算子的函数
def func(x,y,sigma=1):
        return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))
def edge_detect(markpath,savepath):
    
    # 生成标准差为5的5*5高斯算子
    suanzi1 = np.fromfunction(func,(5,5),sigma=5)

    # Laplace扩展算子
    suanzi2 = np.array([[1, 1, 1],
                        [1,-8, 1],
                        [1, 1, 1]])

    # 打开图像并转化成灰度图像
    image = Image.open(markpath).convert("L")
    image_array = np.array(image)

    # 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
    image_blur = signal.convolve2d(image_array, suanzi1, mode="same")

    # 对平滑后的图像进行边缘检测
    image2 = signal.convolve2d(image_array, suanzi2, mode="same")

    # 结果转化到0-255
    image2 = (image2/float(image2.max()))*255
    image2=Image.fromarray(image2).convert('L')
    image2.save(savepath)
    plt.imshow(image2,cmap=cm.gray)
    #plt.savefig(savepath)
    # 将大于灰度平均值的灰度值变成255（白色），便于观察边缘
    #image2[image2>image2.mean()] = 255

    # 显示图像
    # plt.subplot(2,1,1)
    # plt.imshow(image_array,cmap=cm.gray)
    # plt.axis("off")
    # plt.subplot(2,1,2)
    plt.imshow(image2,cmap=cm.gray)
    # plt.axis("off")
    # plt.show()
    #plt.savefig(savepath)
def MedianFilter(src, dst, k = 3, padding = None):
 
	imarray = np.array(Image.open(src))
	height, width = imarray.shape
 
	if not padding:
		edge = int((k-1)/2)
		if height - 1 - edge <= edge or width - 1 - edge <= edge:
			print("The parameter k is to large.")
			return None
		new_arr = np.zeros((height, width), dtype = "uint8")
		for i in range(height):
			for j in range(width):
				if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
					new_arr[i, j] = imarray[i, j]
				else:
					new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])
		new_im = Image.fromarray(new_arr)
		new_im.save(dst)

def closing_algorithm(imgpath,markpath):
    # img=Image.open(imgpath)
    # img=np.asarray(img,dtype=np.uint8)
    # img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img=cv2.imread(imgpath)
    kernal=np.ones((3,3),np.uint8)#(5,5)
    img_close=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernal)
    cv2.imwrite(markpath,img_close)
def medainblur(imgpath,markpath):
    img=cv2.imread(imgpath)
    #img=cv2.cvtColor(np.asarray(img,dtype=np.uint8),cv2.COLOR_RGB2BGR)# 需要确认输入图片的mode
    img_medain=cv2.medianBlur(img,1)#5
    cv2.imwrite(markpath,img_medain)
if __name__=='__main__':
    rootpath='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Dynamic_identify/data/'
    get_video(rootpath)
    get_all_frame_max_mask(rootpath)
    markpath='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Dynamic_identify/data/max_frame.jpg'
    #markpath='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Dynamic_identify/data/max_frame_edge_detect_noslide.jpg'
    savepath=os.path.join(rootpath,'max_frame_canny_150_255.jpg')
    #os.path.join(rootpath,'max_frame_edge_detect_canny_outline.jpg')
    get_outline(markpath,savepath)
    savepath=os.path.join(rootpath,'max_frame_edge_detect_noslide.jpg')
    #edge_detect(markpath,savepath)
    imgpath=os.path.join(rootpath,'max_frame_canny_150_255.jpg')
    markpath=os.path.join(rootpath,'max_frame_canny_150_255_mediam.jpg')
    medainblur(imgpath,markpath)
    src = imgpath
    dst = os.path.join(rootpath,'max_frame_canny_150_255_mediam1.jpg')
    MedianFilter(src, dst)

    # imgpath=os.path.join(rootpath,'max_frame_canny_150_255.jpg')
    # markpath=os.path.join(rootpath,'max_frame_canny_150_255_close.jpg')
    # closing_algorithm(imgpath,markpath)
    # imgpath=markpath
    # markpath=os.path.join(rootpath,'max_frame_canny_150_255_closefillhole.jpg')
    # fillHole(imgpath,markpath)
    # imgpath='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Dynamic_identify/data/max_frame.jpg'
    # savepath=os.path.join(rootpath,'cell_use_threshold.jpg')
    #use_threshold_getcell(imgpath,savepath)