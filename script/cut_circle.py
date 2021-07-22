import cv2
def extract_foreground_mask(img, threshold=0.75, dilate_kernel=2):
    """
    Func: Get a gray image from slide

    Args: img

    Returns:gray_t

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_kernel, dilate_kernel))
    # Convert color space
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray_t = cv2.threshold(gray, threshold * 255, 255,
                                cv2.THRESH_BINARY_INV)
    gray_t = cv2.dilate(gray_t, kernel)
    ret, gray_t = cv2.threshold(gray_t, threshold * 255, 255,
                                cv2.THRESH_BINARY)
    return gray_t

def cut(imgpath,savepath=''):
    rgb_image=cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2RGB)
    #rgb_image = rgb_image[:, :, 0:3]
    gray_image = extract_foreground_mask(rgb_image)
    _,contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#_, 
    bboxes_list = []
    #save outline of countours
    #image=cv2.imread("/mnt/4TB/users/yxzhang/ki67/data/preanno_data/fg/fgmask_2925;A8;KI-67_1050448.svs.png")
    areas={}
    for i in range(len(contours)):
        area_ = cv2.contourArea(contours[i])
        Xs = contours[i][:, 0, 0]
        Ys = contours[i][:, 0, 1]
        x = int(min(Xs))
        y = int(min(Ys))
        w = int((max(Xs)) -x)
        h = int((max(Ys)) -y)
        print(area_)
        if area_ in areas:
            areas[area_]+=1
        else:
            areas[area_]=0
        #if area_>100:
        #cv2.drawContours(image,[contours[i]],-1,(0,255,0),1)
        cv2.rectangle(rgb_image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imwrite('/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/cut_circle/out.jpg',rgb_image)
if __name__=='__main__':
    imgpath='/home/gengxiaoqi/mmdet_singularity/mmdetection_zyx/experiment_gengxiaoqi/Her2/cut_circle/circle.jpg'
    #savepath='',savepath
    cut(imgpath)
