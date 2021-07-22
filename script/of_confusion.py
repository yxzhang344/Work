import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import xlwt
import numpy as np
import os
import math
def symlink_AIS(test_result,savepath):
    clas=['AAH','AIS','BG','LPA','MIA']
    tilespath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/512px_Tiled_all/LPA'
    with open(test_result,'r') as f:
        lines=f.readlines()
        #label=['1','2','3','4','5']
        label=['3','1','2','5','4']
        
        p_true=[]
        p_false=[]
        p_other={}
        # y_true=[]
        # y_pred=[]
        finish_tile=[]
        for line in lines:
            tile_id=line.split('.dat')[0]
            if not (tile_id in finish_tile):
                finish_tile.append(tile_id)
                firstname='_'.join([tile_id.split('_')[1],tile_id.split('_')[2]])
                lastname='_'.join([tile_id.split('_')[3],tile_id.split('_')[4]])
                prestr=line.split('[')[1].split(']')[0].split()
                pre=[float(pre) for pre in prestr]
                p_pre=max(pre)
                y_pred=str(pre.index(max(pre)))
                y_true=line.split('labels: \t')[1].strip('\n')
                from_=os.path.join(tilespath,firstname,'5.0',lastname+'.jpeg')
                if y_true=='4':
                    to_=os.path.join(savepath,clas[int(y_pred)-1],str(round(p_pre,2))+'_'+firstname+'_'+lastname+'.jpeg')
                    if not os.path.exists(os.path.join(savepath,clas[int(y_pred)-1])):
                        os.makedirs(os.path.join(savepath,clas[int(y_pred)-1]))
                    print(from_,to_)
                    os.symlink(from_,to_)
def bar(test_result,savepath):
    clas=['BG','AAH','AIS','MIA','LPA']
    with open(test_result,'r') as f:
        lines=f.readlines()
        #label=['1','2','3','4','5']
        label=['3','1','2','5','4']
        for i in range(len(label)):
            TP=[0 for i in range(20)]
            FP=[0 for i in range(20)] 
            finish_tile=[]
            for line in lines:
                tile_id=line.split('.dat')[0]
                if not (tile_id in finish_tile):
                    finish_tile.append(tile_id)
                    prestr=line.split('[')[1].split(']')[0].split()
                    pre=[float(pre) for pre in prestr]
                    p_pre=max(pre)
                    p_index=math.floor(p_pre/0.05)
                    y_pred=str(pre.index(max(pre)))
                    y_true=line.split('labels: \t')[1].strip('\n')
                    if y_pred==label[i]:
                        if y_true==label[i]:
                            TP[p_index]+=1
                        else: 
                            FP[p_index]+=1
            x=[0.05*i for i in range(20)]
            FP=[-p for p in FP]
            plt.figure()
            bar_width=0.05
            plt.bar(x,TP,bar_width,align='edge',label='TP')
            plt.bar(x,FP,bar_width,align='edge',label='FP')
            plt.title(clas[i]+' TP and FP Histogram')
            plt.legend()
            plt.savefig(savepath+clas[i]+'.jpeg')

def hist(test_result,savepath):
    clas=['BG','AAH','AIS','MIA','LPA']
    with open(test_result,'r') as f:
        lines=f.readlines()
        #label=['1','2','3','4','5']
        label=['3','1','2','5','4']
        for i in range(len(label)):
            p_true=[]
            p_false=[]
            p_other={}
            # y_true=[]
            # y_pred=[]
            finish_tile=[]
            for line in lines:
                tile_id=line.split('.dat')[0]
                if not (tile_id in finish_tile):
                    finish_tile.append(tile_id)
                    prestr=line.split('[')[1].split(']')[0].split()
                    pre=[float(pre) for pre in prestr]
                    p_pre=max(pre)
                    y_pred=str(pre.index(max(pre)))
                    y_true=line.split('labels: \t')[1].strip('\n')
                    if y_true==label[i]:
                        if y_pred==label[i]:
                            p_true.append(p_pre)
                        else:
                            p_false.append(pre[int(y_true)])
                            if not (y_pred in p_other):
                                p_other[y_pred]=[]
                                p_other[y_pred].append(p_pre)
                            else:
                                p_other[y_pred].append(p_pre)
                    else:
                        continue
            other=len(p_other)
            if other==0:
                sub=[121,122]
            elif other==1:
                sub=[221,222,223]
            elif other==2:
                sub=[221,222,223,224]
            elif other==3:
                sub=[321,322,323,324,325]
            elif other==4:
                sub=[321,322,323,324,325,326]
            print(len(sub))
            plt.figure(i)
            plt.subplot(sub[0])
            plt.hist(x=p_true,bins=10)
            plt.title(clas[i]+'  True')
            #plt.xlabel('P')
            plt.subplot(sub[1])
            plt.hist(x=p_false,bins=10)
            plt.title(clas[i]+'  False')
            
            for s in range(2,len(sub)):
                
                #plt.title('Other Probability Histogram')
                #plt.ylabel()
                if s%2==0:
                    plt.subplot(sub[s])
                    plt.hist(x=p_other[list(p_other.keys())[s-2]],bins=10)
                    plt.ylabel(clas[label.index(list(p_other.keys())[s-2])])
                else:
                    ax=plt.subplot(sub[s])
                    plt.hist(x=p_other[list(p_other.keys())[s-2]],bins=10)
                    ax.yaxis.tick_right()
                    plt.ylabel(clas[label.index(list(p_other.keys())[s-2])])
            plt.savefig(savepath+label[i]+'.png')


def accuracy_revall(test_result,savepath):
    #clas=['AAH','AIS','BG','LPA','MIA']
    clas=['BG','AAH','AIS','MIA','LPA']
    with open(test_result,'r') as f:
        #label=['1','2','3','4','5']
        label=['3','1','2','5','4']
        y_true=[]
        y_pred=[]
        finish_tile=[]
        for line in f:
            tile_id=line.split('.dat')[0]
            if not (tile_id in finish_tile):
                finish_tile.append(tile_id)
                prestr=line.split('[')[1].split(']')[0].split()
                pre=[float(pre) for pre in prestr]
                y_pred.append(str(pre.index(max(pre))))
                y_true.append(line.split('labels: \t')[1].strip('\n'))
        TP={la:0 for la in label}
        FP={la:0 for la in label}
        TN={la:0 for la in label}
        FN={la:0 for la in label}
        for la in label:
            for i in range(len(y_true)):
                if y_true[i]==la:
                    if y_pred[i]==la:
                        TP[la]+=1
                    else:
                        FN[la]+=1
                elif y_pred[i]==la:
                    FP[la]+=1
                else:
                    TN[la]+=1
        accuracy={la:0 for la in label}
        precision={la:0 for la in label}
        recall={la:0 for la in label}
        specificity={la:0 for la in label}
        f1={la:0 for la in label}
        for la in label:
            accuracy[la]=round((TP[la]+TN[la])/(TP[la]+TN[la]+FP[la]+FN[la]),2)
            precision[la]=round(TP[la]/(TP[la]+FP[la]),2)
            recall[la]=round(TP[la]/(TP[la]+FN[la]),2)
            specificity[la]=round(TN[la]/(TN[la]+FP[la]),2)
            f1[la]=round(2*precision[la]*recall[la]/(precision[la]+recall[la]),2)
        
        workbook=xlwt.Workbook()
        worksheet=workbook.add_sheet('sheet')
        for i in range(len(label)):
            worksheet.write(0,i+1,label=clas[i])
            worksheet.write(1,i+1,label=accuracy[label[i]])
            worksheet.write(2,i+1,label=precision[label[i]])
            worksheet.write(3,i+1,label=recall[label[i]])
            worksheet.write(4,i+1,label=specificity[label[i]])
            worksheet.write(5,i+1,label=f1[label[i]])
        name=['accuracy','precision','recall','specificity','f1']
        for i in range(5):
            worksheet.write(i+1,0,label=name[i])
        workbook.save(savepath)
def paper_result_table(test_result,savepath='',model=''):
    #clas=['AAH','AIS','BG','LPA','MIA']
    clas=['BG','Acinar','LPA','Micropapillary','Papillary','Solid']#['BG','AAH','AIS','MIA','LPA']
    with open(test_result,'r') as f:
        #label=['1','2','3','4','5']
        label=['2','1','3','4','5','6']
        #['2','1','3','4','5','6']
        #label=['3','1','2','5','4']
        y_true=[]
        y_pred=[]
        p_pre=[]
        finish_tile=[]
        for line in f:
            tile_id=os.path.splitext(line)[0]#.split('.dat')
            if not (tile_id in finish_tile):
                finish_tile.append(tile_id)
                prestr=line.split('[')[1].split(']')[0].split()
                pre=[float(pr) for pr in prestr]
                p_pre.append(pre)
                y_pred.append(str(pre.index(max(pre))))#1-5
                y_true.append(line.split('labels: \t')[1].strip('\n'))#1-5
        TP={la:0 for la in label}
        FP={la:0 for la in label}
        TN={la:0 for la in label}
        FN={la:0 for la in label}
        for la in label:
            for i in range(len(y_true)):
                if y_true[i]==la:
                    if y_pred[i]==la:
                        TP[la]+=1
                    else:
                        FN[la]+=1
                elif y_pred[i]==la:
                    FP[la]+=1
                else:
                    TN[la]+=1
        #accuracy={la:0 for la in label}
        precision=[]
        recall=[]
        f1=[]
        for la in label:
            #accuracy[la]=round((TP[la]+TN[la])/(TP[la]+TN[la]+FP[la]+FN[la]),2)
            #print(la)
            precision.append(TP[la]/(TP[la]+FP[la]+0.00001))
            recall.append(TP[la]/(TP[la]+FN[la]))
            l=label.index(la)
            f1.append(2*precision[l]*recall[l]/(precision[l]+recall[l]+0.00001))
        
        root=os.path.dirname(test_result)
        '''根据test时得到的文件名中的AUC计算'''
        # auc_=[]
        # for fl in os.listdir(root):
        #     if ('out1_roc_data_AvPb' in fl) and not ('micro' in fl) and not ('macro' in fl):
        #         auc_.append(float(fl.split('_')[5]))
        # AUC=round(np.mean(auc_),2)
        '''感觉test时tiles有重复，结果可能不准，下面去重再计算，与上面比较下
        发现与上面基本一致，fpr tpr量级也差不多，就用文件得到的'''
        auc_m=[]
        print('num of tiles is ',len(y_true))
        for l in label:
            y_real=[]
            y_p=[]
            for i in range(len(y_pred)):
                if y_true[i]==l:
                    y_real.append(1)
                else:
                    y_real.append(0)
                y_p.append(p_pre[i][int(l)])
            fpr, tpr, thresholds = roc_curve(y_real, y_p)#此处每次为何是3各值，不应该是10个吗，为何结果有nan?
            print(len(fpr), len(tpr))
            auc_m.append(auc(fpr, tpr))
        print(auc_m)
        AUC=round(np.mean(auc_m),2)
        '''
        workbook=xlwt.Workbook()
        worksheet=workbook.add_sheet('sheet')
        name=['Precision','Recall','F1','AUC']#,'auc_m'
        for i in range(4):
            worksheet.write(0,i+1,label=name[i])
        worksheet.write(1,0,label=model)
        worksheet.write(1,1,label=str(round(np.mean(precision),2)))
        worksheet.write(1,2,label=str(round(np.mean(recall),2)))
        worksheet.write(1,3,label=str(round(np.mean(f1),2)))
        worksheet.write(1,4,label=str(AUC))
        '''
        #worksheet.write(1,5,label=str(round(np.mean(auc_m),2)))
        
        #workbook.save(savepath)
        return precision,recall,f1,AUC
def paper_bagging(test_result,savepath='',model=''):
    y_true={}
    p_pred={}
    y_pred=[]
    tile_n={}
    N=len(os.listdir(test_result))
    for t_id in os.listdir(test_result):
        for fi in os.listdir(os.path.join(test_result,t_id)):
            if 'test' in fi:
                root=os.path.join(test_result,t_id,fi)
                state=os.path.join(root,'out_filename_Stats.txt')
                with open(state,'r') as f:
                    #label=['1','2','3','4','5']
                    label=['3','1','2','5','4']
                    
                    finish_tile=[]
                    for line in f:
                        tile_id=line.split('.dat')[0]
                        if not (tile_id in finish_tile):
                            finish_tile.append(tile_id)
                            prestr=line.split('[')[1].split(']')[0].split()
                            pre=[float(pre) for pre in prestr]
                            if not (tile_id in p_pred):
                                p_pred[tile_id]=pre
                                tile_n[tile_id]=1
                            else:
                                tile_n[tile_id]+=1
                                p_pred[tile_id]=[p_pred[tile_id][c]+pre[c] for c in range(len(pre))]
                            if not (tile_id in y_true):
                                y_true[tile_id]=line.split('labels: \t')[1].strip('\n')
                break
    for n in tile_n:
        if tile_n[n]!=N:
            p_pred.pop(n)
            y_true.pop(n)
    y_true_=[y_true[i] for i in y_true]
    for k in p_pred:
        y_pred.append(str(p_pred[k].index(max(p_pred[k]))))
    TP={la:0 for la in label}
    FP={la:0 for la in label}
    TN={la:0 for la in label}
    FN={la:0 for la in label}
    for la in label:
        for i in range(len(y_true_)):
            if y_true_[i]==la:
                if y_pred[i]==la:
                    TP[la]+=1
                else:
                    FN[la]+=1
            elif y_pred[i]==la:
                FP[la]+=1
            else:
                TN[la]+=1
    #accuracy={la:0 for la in label}
    precision=[]
    recall=[]
    f1=[]
    for la in label:
        #accuracy[la]=round((TP[la]+TN[la])/(TP[la]+TN[la]+FP[la]+FN[la]),2)
        precision.append(TP[la]/(TP[la]+FP[la]))
        recall.append(TP[la]/(TP[la]+FN[la]))
        l=label.index(la)
        f1.append(2*precision[l]*recall[l]/(precision[l]+recall[l]))
    
    #root=os.path.dirname(test_result)
    '''根据test时得到的文件名中的AUC计算'''
    # auc_=[]
    # for fl in os.listdir(root):
    #     if ('out1_roc_data_AvPb' in fl) and not ('micro' in fl) and not ('macro' in fl):
    #         auc_.append(float(fl.split('_')[5]))
    # AUC=round(np.mean(auc_),2)
    '''感觉test时tiles有重复，结果可能不准，下面去重再计算，与上面比较下
    发现与上面基本一致，fpr tpr量级也差不多，就用文件得到的'''
    auc_m=[]
    #print('num of tiles is ',len(y_true))
    for l in label:
        y_real=[]
        y_p=[]
        print(len(y_pred),len(y_true),len(p_pred))#,tile_n
        for i in range(len(y_pred)):
            if y_true_[i]==l:
                y_real.append(1)
            else:
                y_real.append(0)
            y_p.append(p_pred[list(p_pred.keys())[i]][int(l)]/N)
        fpr, tpr, thresholds = roc_curve(y_real, y_p)#此处每次为何是3各值，不应该是10个吗，为何结果有nan?
        print(len(fpr), len(tpr))
        auc_m.append(auc(fpr, tpr))
    print(auc_m)
    '''
    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('sheet')
    name=['Precision','Recall','F1','AUC']#,'auc_m'
    for i in range(4):
        worksheet.write(0,i+1,label=name[i])
    worksheet.write(1,0,label=model)
    worksheet.write(1,1,label=str(round(np.mean(precision),2)))
    worksheet.write(1,2,label=str(round(np.mean(recall),2)))
    worksheet.write(1,3,label=str(round(np.mean(f1),2)))
    #worksheet.write(1,4,label=str(AUC))
    worksheet.write(1,4,label=str(round(np.mean(auc_m),2)))
    
    workbook.save(savepath)'''
    return precision,recall,f1,auc_m
def paper_bagging_xinjiang(test_result,model_name,savepath='',model='',output_dir=''):
    y_true={}
    p_pred={}
    y_pred=[]
    tile_n={}
    N=len(os.listdir(test_result))
    for t_id in os.listdir(test_result):
        for fi in os.listdir(os.path.join(test_result,t_id,model_name,'test')):
            if 'test' in fi:
                root=os.path.join(test_result,t_id,model_name,'test',fi)
                state=os.path.join(root,'out_filename_Stats.txt')
                with open(state,'r') as f:
                    #label=['1','2','3','4','5']
                    #label=['3','1','2','5','4']
                    label=['2','1','3','4','5','6']
                    finish_tile=[]
                    for line in f:
                        tile_id=line.split('.dat')[0]+'_'+line.split('labels: \t')[-1]#不能改为加入
                        if not (tile_id in finish_tile):
                            finish_tile.append(tile_id)
                            prestr=line.split('[')[1].split(']')[0].split()
                            pre=[float(pre) for pre in prestr]
                            if not (tile_id in p_pred):
                                p_pred[tile_id]=pre
                                tile_n[tile_id]=1
                            else:
                                tile_n[tile_id]+=1
                                p_pred[tile_id]=[p_pred[tile_id][c]+pre[c] for c in range(len(pre))]
                            if not (tile_id in y_true):
                                y_true[tile_id]=line.split('labels: \t')[1].strip('\n')#注意：6out的真实label给的是1，2，3，4，5,6
                break
    #print(p_pred)
    for n in tile_n:
        #print(tile_n[n])
        if tile_n[n]!=N:
            p_pred.pop(n)
            y_true.pop(n)
    y_true_=[y_true[i] for i in y_true]#[1,3,2,4....]
    for k in p_pred:
        y_pred.append(str(p_pred[k].index(max(p_pred[k]))+1))#6out has +1
        #y_pred.append(str(p_pred[k].index(max(p_pred[k]))))
    '''calculate roc'''
    if output_dir!='':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fpr = dict()#{0: array([nan, nan, nan]), 1: array([0. , 0.1, 1. ]), 2: array([0. , 0.1, 1. ]), 3: array([0. , 0.1, 1. ]), 4: a
        tpr = dict()
        thresholds=dict()
        roc_auc=dict()
        y_true_onehot=[]
        for y in y_true_:
            onehot=[0 for i in range(len(label))]
            onehot[int(y)-1]=1#6out using 
            #onehot[int(y)]=1
            y_true_onehot.append(onehot)
        #p_pred_=[[p_pred[j][i]/N for i in range(len(p_pred[j]))] for j in p_pred]
        p_pred_=[[p_pred[j][i]/N for i in range(len(p_pred[j])) if i!=0] for j in p_pred]#6out using  if i!=0
        #print(p_pred)
        n_classes=len(p_pred_[0])
        y_pre=np.array(p_pred_)
        y_score=np.array(y_true_onehot)
        #print(y_pre[0],y_score[0])

        for i in range(n_classes):
            #print(y_ref[:, i], y_score[:, i])
            fpr[i], tpr[i], thresholds[i] = roc_curve(y_score[:, i], y_pre[:, i])#此处每次为何是3各值，不应该是10个吗，为何结果有nan?
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(n_classes):
            output = open(os.path.join(output_dir,'_roc_data_AvPb_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc[i]) + '.txt'),'w')
            for kk in range(len(tpr[i])):
                output.write("%f\t%f\n" % (fpr[i][kk], tpr[i][kk]) )
            output.close()
    '''calculate bagging precisice recall and f1'''
    TP={la:0 for la in label}
    FP={la:0 for la in label}
    TN={la:0 for la in label}
    FN={la:0 for la in label}
    for la in label:
        for i in range(len(y_true_)):
            if y_true_[i]==la:
                if y_pred[i]==la:
                    TP[la]+=1
                else:
                    FN[la]+=1
            elif y_pred[i]==la:
                FP[la]+=1
            else:
                TN[la]+=1
    #accuracy={la:0 for la in label}
    precision=[]
    recall=[]
    f1=[]
    for la in label:
        #accuracy[la]=round((TP[la]+TN[la])/(TP[la]+TN[la]+FP[la]+FN[la]),2)
        precision.append(TP[la]/(TP[la]+FP[la]+0.00001))
        recall.append(TP[la]/(TP[la]+FN[la]+0.00001))
        l=label.index(la)
        f1.append(2*precision[l]*recall[l]/(precision[l]+recall[l]+0.00001))
    
    #root=os.path.dirname(test_result)
    '''根据test时得到的文件名中的AUC计算'''
    # auc_=[]
    # for fl in os.listdir(root):
    #     if ('out1_roc_data_AvPb' in fl) and not ('micro' in fl) and not ('macro' in fl):
    #         auc_.append(float(fl.split('_')[5]))
    # AUC=round(np.mean(auc_),2)
    '''感觉test时tiles有重复，结果可能不准，下面去重再计算，与上面比较下
    发现与上面基本一致，fpr tpr量级也差不多，就用文件得到的'''
    auc_m=[]
    #print('num of tiles is ',len(y_true))
    for l in label:
        y_real=[]
        y_p=[]
        #print(len(y_pred),len(y_true),len(p_pred))#,tile_n
        for i in range(len(y_pred)):
            #6out using y_true_[i]==str(int(l)-1)  其他直接y_true_[i]==l
            if y_true_[i]==l:
                y_real.append(1)
            else:
                y_real.append(0)
            y_p.append(p_pred[list(p_pred.keys())[i]][int(l)]/N)#6out nohas -1
        fpr, tpr, thresholds = roc_curve(y_real, y_p)#此处每次为何是3各值，不应该是10个吗，为何结果有nan?
        #print(len(fpr), len(tpr))
        auc_m.append(auc(fpr, tpr))
    # print(l,y_real, y_p)
    # print(fpr, tpr)
    print(auc_m)
    return precision,recall,f1,auc_m

def paper_bagging_xinjiang_2modelmix(test_list,model_name,savepath='',model='',output_dir=''):
    '''bagging 的子实验来自两个模型'''
    y_true={}
    p_pred={}
    y_pred=[]
    tile_n={}
    N=len(test_list)
    for t_id in test_list[:7]:
        for fi in os.listdir(os.path.join(t_id,model_name[0],'test')):
            if 'test' in fi:
                root=os.path.join(test_result,t_id,model_name[0],'test',fi)
                state=os.path.join(root,'out_filename_Stats.txt')
                with open(state,'r') as f:
                    #label=['1','2','3','4','5']
                    #label=['3','1','2','5','4']
                    label=['2','1','3','4','5','6']
                    finish_tile=[]
                    for line in f:
                        tile_id=line.split('.dat')[0]+'_'+line.split('labels: \t')[-1]#不能改为加入
                        if not (tile_id in finish_tile):
                            finish_tile.append(tile_id)
                            prestr=line.split('[')[1].split(']')[0].split()
                            pre=[float(pre) for pre in prestr]
                            if not (tile_id in p_pred):
                                p_pred[tile_id]=pre
                                tile_n[tile_id]=1
                            else:
                                tile_n[tile_id]+=1
                                p_pred[tile_id]=[p_pred[tile_id][c]+pre[c] for c in range(len(pre))]
                            if not (tile_id in y_true):
                                y_true[tile_id]=line.split('labels: \t')[1].strip('\n')#注意：6out的真实label给的是0，1，2，3，4，5
                break
    for t_id in test_list[7:]:
        for fi in os.listdir(os.path.join(t_id,model_name[1],'test')):
            if 'test' in fi:
                root=os.path.join(test_result,t_id,model_name[1],'test',fi)
                state=os.path.join(root,'out_filename_Stats.txt')
                with open(state,'r') as f:
                    #label=['1','2','3','4','5']
                    #label=['3','1','2','5','4']
                    label=['2','1','3','4','5','6']
                    finish_tile=[]
                    for line in f:
                        tile_id=line.split('.dat')[0]+'_'+line.split('labels: \t')[-1]#不能改为加入
                        if not (tile_id in finish_tile):
                            finish_tile.append(tile_id)
                            prestr=line.split('[')[1].split(']')[0].split()
                            pre=[float(pre) for pre in prestr]
                            if not (tile_id in p_pred):
                                p_pred[tile_id]=pre
                                tile_n[tile_id]=1
                            else:
                                tile_n[tile_id]+=1
                                p_pred[tile_id]=[p_pred[tile_id][c]+pre[c] for c in range(len(pre))]
                            if not (tile_id in y_true):
                                y_true[tile_id]=line.split('labels: \t')[1].strip('\n')#注意：6out的真实label给的是0，1，2，3，4，5
                break
    #print(p_pred)
    for n in tile_n:
        print(tile_n[n])
        if tile_n[n]!=N:
            p_pred.pop(n)
            y_true.pop(n)
    y_true_=[y_true[i] for i in y_true]#[1,3,2,4....]
    for k in p_pred:
        y_pred.append(str(p_pred[k].index(max(p_pred[k]))))#6out has +1
    '''calculate roc'''
    if output_dir!='':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fpr = dict()#{0: array([nan, nan, nan]), 1: array([0. , 0.1, 1. ]), 2: array([0. , 0.1, 1. ]), 3: array([0. , 0.1, 1. ]), 4: a
        tpr = dict()
        thresholds=dict()
        roc_auc=dict()
        y_true_onehot=[]
        for y in y_true_:
            onehot=[0 for i in range(len(label))]
            onehot[int(y)]=1#6out nousing -1
            y_true_onehot.append(onehot)
        p_pred_=[[p_pred[j][i]/N for i in range(len(p_pred[j]))] for j in p_pred]#6out nousing  if i!=0
        #print(p_pred)
        n_classes=len(p_pred_[0])
        y_pre=np.array(p_pred_)
        y_score=np.array(y_true_onehot)
        print(y_pre[0],y_score[0])

        for i in range(n_classes):
            #print(y_ref[:, i], y_score[:, i])
            fpr[i], tpr[i], thresholds[i] = roc_curve(y_score[:, i], y_pre[:, i])#此处每次为何是3各值，不应该是10个吗，为何结果有nan?
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(n_classes):
            output = open(os.path.join(output_dir,'_roc_data_AvPb_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc[i]) + '.txt'),'w')
            for kk in range(len(tpr[i])):
                output.write("%f\t%f\n" % (fpr[i][kk], tpr[i][kk]) )
            output.close()
    '''calculate bagging precisice recall and f1'''
    TP={la:0 for la in label}
    FP={la:0 for la in label}
    TN={la:0 for la in label}
    FN={la:0 for la in label}
    for la in label:
        for i in range(len(y_true_)):
            if y_true_[i]==la:
                if y_pred[i]==la:
                    TP[la]+=1
                else:
                    FN[la]+=1
            elif y_pred[i]==la:
                FP[la]+=1
            else:
                TN[la]+=1
    #accuracy={la:0 for la in label}
    precision=[]
    recall=[]
    f1=[]
    for la in label:
        #accuracy[la]=round((TP[la]+TN[la])/(TP[la]+TN[la]+FP[la]+FN[la]),2)
        precision.append(TP[la]/(TP[la]+FP[la]+0.00001))
        recall.append(TP[la]/(TP[la]+FN[la]+0.00001))
        l=label.index(la)
        f1.append(2*precision[l]*recall[l]/(precision[l]+recall[l]+0.00001))
    
    #root=os.path.dirname(test_result)
    '''根据test时得到的文件名中的AUC计算'''
    # auc_=[]
    # for fl in os.listdir(root):
    #     if ('out1_roc_data_AvPb' in fl) and not ('micro' in fl) and not ('macro' in fl):
    #         auc_.append(float(fl.split('_')[5]))
    # AUC=round(np.mean(auc_),2)
    '''感觉test时tiles有重复，结果可能不准，下面去重再计算，与上面比较下
    发现与上面基本一致，fpr tpr量级也差不多，就用文件得到的'''
    auc_m=[]
    #print('num of tiles is ',len(y_true))
    for l in label:
        y_real=[]
        y_p=[]
        #print(len(y_pred),len(y_true),len(p_pred))#,tile_n
        for i in range(len(y_pred)):
            #6out using y_true_[i]==str(int(l)-1)  其他直接y_true_[i]==l
            if y_true_[i]==l:
                y_real.append(1)
            else:
                y_real.append(0)
            y_p.append(p_pred[list(p_pred.keys())[i]][int(l)]/N)#6out has -1
        fpr, tpr, thresholds = roc_curve(y_real, y_p)#此处每次为何是3各值，不应该是10个吗，为何结果有nan?
        #print(len(fpr), len(tpr))
        auc_m.append(auc(fpr, tpr))
    # print(l,y_real, y_p)
    # print(fpr, tpr)
    print('precision is {} recall is {} f1 is {} auc_m is '.format(str(round(np.mean(precision),2)),str(round(np.mean(recall),2)),str(round(np.mean(f1),2)),str(round(np.mean(auc_m),2))))
    print(auc_m)
    return precision,recall,f1,auc_m

def paper(result_list,savepath,baggingpath=[]):
    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('sheet')
    name=['Precision','Recall','F1','AUC']#,'auc_m'
    for i in range(4):
        worksheet.write(0,i+1,label=name[i])
    for j in range(len(result_list)):
        print(result_list[j])
        precision,recall,f1,AUC=paper_result_table(result_list[j],savepath='',model='')
        worksheet.write(j+1,0,label=result_list[j])
        worksheet.write(j+1,1,label=str(round(np.mean(precision),2)))
        worksheet.write(j+1,2,label=str(round(np.mean(recall),2)))
        worksheet.write(j+1,3,label=str(round(np.mean(f1),2)))
        worksheet.write(j+1,4,label=str(AUC))
        #worksheet.write(1,5,label=str(round(np.mean(auc_m),2)))
    j+=1
    if not baggingpath==[]:
        for b in range(len(baggingpath)):
            precision,recall,f1,auc_m=paper_bagging(baggingpath[b],savepath='',model='')
            worksheet.write(j+b+1,0,label=baggingpath[b])
            worksheet.write(j+b+1,1,label=str(round(np.mean(precision),2)))
            worksheet.write(j+b+1,2,label=str(round(np.mean(recall),2)))
            worksheet.write(j+b+1,3,label=str(round(np.mean(f1),2)))
            worksheet.write(j+b+1,4,label=str(round(np.mean(auc_m),2)))

    workbook.save(savepath)
    
def paper_xinjiang(result_list,savepath,model_list,baggingpath=[]):
    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('sheet')
    name=['Precision','Recall','F1','AUC']#,'auc_m'
    for i in range(4):
        worksheet.write(0,i+1,label=name[i])
    for j in range(len(result_list)):
        print(result_list[j])
        precision,recall,f1,AUC=paper_result_table(result_list[j],savepath='',model='')
        worksheet.write(j+1,0,label=result_list[j])#.split('bagging_patient_level')[1]
        worksheet.write(j+1,1,label=str(round(np.mean(precision),2)))
        worksheet.write(j+1,2,label=str(round(np.mean(recall),2)))
        worksheet.write(j+1,3,label=str(round(np.mean(f1),2)))
        worksheet.write(j+1,4,label=str(AUC))
        #worksheet.write(1,5,label=str(round(np.mean(auc_m),2)))
    j+=1
    if not baggingpath==[]:
        for b in range(len(model_list)):
            mod_n=model_list[b]
            precision,recall,f1,auc_m=paper_bagging_xinjiang(baggingpath[0],mod_n,savepath='',model='')
            worksheet.write(j+b+1,0,label=mod_n)
            worksheet.write(j+b+1,1,label=str(round(np.mean(precision),2)))
            worksheet.write(j+b+1,2,label=str(round(np.mean(recall),2)))
            worksheet.write(j+b+1,3,label=str(round(np.mean(f1),2)))
            worksheet.write(j+b+1,4,label=str(round(np.mean(auc_m),2)))

    workbook.save(savepath)

def confusion_mx(test_result,save_path,name):
    clas=['BG','Acinar','LPA','Micropapillary','Papillary','Solid']
    #clas=['BG','AAH','AIS','MIA','LPA']
    with open(test_result,'r') as f:
        y_true=[]
        y_pred=[]
        finish_tile=[]
        for line in f:
            tile_id=line.split('.dat')[0]
            if not (tile_id in finish_tile):
                finish_tile.append(tile_id)
                prestr=line.split('[')[1].split(']')[0].split()
                pre=[float(pre) for pre in prestr]
                y_pred.append(str(pre.index(max(pre))))
                y_true.append(line.split('labels: \t')[1].strip('\n'))
                
    #print(y_pred,y_true)
    c2=confusion_matrix(y_true,y_pred,labels=['2','1','3','4','5','6'])
    #['3','1','2','5','4']
    #cm_normalized = c2.astype('float') / c2.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,8))#figsize=(20,16)
    ax = sns.heatmap(c2, xticklabels=clas, yticklabels=clas, 
                    linewidths=0.2, cmap="YlGnBu",annot=True)
    plt.title(name+" Set Confusion Matrix")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(save_path)
    
    # fig=sns.heatmap(cm_normalized,annot=True)
    # heatmap=fig.get_figure()
    # heatmap.savefig(save_path,dpi=400)
def acording_confusionmx_get_kappa(test_result):
    clas=['BG','Acinar','LPA','Micropapillary','Papillary','Solid']
    #clas=['BG','AAH','AIS','MIA','LPA']
    with open(test_result,'r') as f:
        y_true=[]
        y_pred=[]
        finish_tile=[]
        for line in f:
            tile_id=line.split('.dat')[0]
            if not (tile_id in finish_tile):
                finish_tile.append(tile_id)
                prestr=line.split('[')[1].split(']')[0].split()
                pre=[float(pre) for pre in prestr]
                y_pred.append(str(pre.index(max(pre))))
                y_true.append(line.split('labels: \t')[1].strip('\n'))
                
    #print(y_pred,y_true)
    c2=confusion_matrix(y_true,y_pred,labels=['2','1','3','4','5','6'])
    #['3','1','2','5','4']
    #print(c2,c2.sum(axis=1),c2.sum(axis=0))
    sum_n=sum(c2.sum(axis=1))
    p0=0
    for i in range(len(c2)):
        p0+=c2[i][i]
    p0_ratio=p0/sum_n
    pe=sum([c2.sum(axis=1)[i]*c2.sum(axis=0)[i] for i in range(len(c2))])
    pe_ratio=pe/sum_n**2
    kappa=(p0_ratio-pe_ratio)/(1-pe_ratio)
    print(kappa)#,pe_ratio,p0_ratio,sum_n,[c2.sum(axis=1)[i]*c2.sum(axis=0)[i] for i in range(len(c2))]

    #cm_normalized = c2.astype('float') / c2.sum(axis=1)[:, np.newaxis]
    # plt.figure(figsize=(10,8))#figsize=(20,16)
    # ax = sns.heatmap(c2, xticklabels=clas, yticklabels=clas, 
    #                 linewidths=0.2, cmap="YlGnBu",annot=True)
    # plt.title(name+" Set Confusion Matrix")
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.savefig(save_path)
    
    # fig=sns.heatmap(cm_normalized,annot=True)
    # heatmap=fig.get_figure()
    # heatmap.savefig(save_path,dpi=400)
def confusion_mx_percent(test_result,save_path,name):
    #sns.set(font_scale=.9)
    clas=['BG','Acinar','LPA','Micropapillary','Papillary','Solid']
    #clas=['AAH','AIS','BG','LPA','MIA']
    #clas=['BG','AAH','AIS','MIA','LPA']
    with open(test_result,'r') as f:
        y_true=[]
        y_pred=[]
        finish_tile=[]
        for line in f:
            tile_id=line.split('.dat')[0]
            if not (tile_id in finish_tile):
                finish_tile.append(tile_id)
                prestr=line.split('[')[1].split(']')[0].split()
                pre=[float(pre) for pre in prestr]
                y_pred.append(str(pre.index(max(pre))))
                y_true.append(line.split('labels: \t')[1].strip('\n'))
                
    #print(y_pred,y_true)
    c2=confusion_matrix(y_true,y_pred,labels=['2','1','3','4','5','6'])#['3','1','2','5','4']
    cm_normalized = c2.astype('float') / c2.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,8))#figsize=(20,16)
    ax = sns.heatmap(cm_normalized, xticklabels=clas, yticklabels=clas, 
                    linewidths=0.2, cmap="YlGnBu",annot=True,annot_kws={"fontsize":12})
    if name=='Test':
        plt.title("Two doctors")#name+" Set Confusion Matrix——One doctor"
    else:
        plt.title("Bagging_"+name+" Set Confusion Matrix")
        #plt.title(name+" Set Confusion Matrix")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(save_path)
def confusion_mx_percent_noBG(test_result,save_path,name):
    #clas=['AAH','AIS','BG','LPA','MIA']
    clas=['AAH','AIS','MIA','LPA']
    with open(test_result,'r') as f:
        y_true=[]
        y_pred=[]
        finish_tile=[]
        for line in f:
            tile_id=line.split('.dat')[0]
            if not (tile_id in finish_tile):
                finish_tile.append(tile_id)
                prestr=line.split('[')[1].split(']')[0].split()
                pre=[float(pre) for pre in prestr]
                y_pred.append(str(pre.index(max(pre))))
                y_true.append(line.split('labels: \t')[1].strip('\n'))
                
    #print(y_pred,y_true)
    c2=confusion_matrix(y_true,y_pred,labels=['1','2','5','4'])
    cm_normalized = c2.astype('float') / c2.sum(axis=1)[:, np.newaxis]
    plt.figure()#figsize=(20,16)
    ax = sns.heatmap(cm_normalized, xticklabels=clas, yticklabels=clas, 
                    linewidths=0.2, cmap="YlGnBu",annot=True)
    plt.title(name+" Set Confusion Matrix——tiles in label")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(save_path)
def dominant_minor_ave_p(test_path,save_path):
    clas=['AAH','AIS','LPA','MIA']
    with open(test_result,'r') as f:
        wsi_result={}
        wsi_count={}
        for line in f:
            prestr=line.split('[')[1].split(']')[0].split()
            pre=[float(pr) for pr in prestr]
            presum=1-pre[0]
            pre_norm=[pr/presum for pr in pre[1:]]
            wsi_id=line.split('_files_')[0].split('test_')[1]
            if wsi_id in  wsi_result:
                for i in range(len(wsi_result[wsi_id])):
                    wsi_result[wsi_id][i]+=pre_norm[i]
                wsi_count[wsi_id]+=1
            else:
                wsi_result[wsi_id]=pre_norm
                wsi_count[wsi_id]=1

    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('predominant_minor')
    worksheet.write(0,0,label='WSI id')
    worksheet.write(0,1,label='tiles num')
    worksheet.write(0,2,label='predict P')
    worksheet.write(0,3,label='predict dominant')
    worksheet.write(0,4,label='predict minor')
    for i in range(len(wsi_result.keys())):
        num=list(wsi_count.values())[i]
        predict_P=list(wsi_result.values())[i]
        norm_P=[round(p/num,3) for p in predict_P]
        norm_P.pop(2)
        dominant=sorted(norm_P)[-1]
        minor=sorted(norm_P)[-2]
        dominant_index=norm_P.index(dominant)
        minor_index=norm_P.index(minor)
        worksheet.write(i+1,0,label=list(wsi_result.keys())[i])
        worksheet.write(i+1,1,label=str(num))
        worksheet.write(i+1,2,label=str(norm_P))
        worksheet.write(i+1,3,label=clas[dominant_index])
        worksheet.write(i+1,4,label=clas[minor_index])
    workbook.save(save_path)
def dominant_minor_clas_num(test_path,save_path):
    clas=['AAH','AIS','BG','LPA','MIA']
    with open(test_result,'r') as f:
        wsi_result={}
        #wsi_count={}
        for line in f:
            prestr=line.split('[')[1].split(']')[0].split()
            pre=[float(pr) for pr in prestr]
            presum=1-pre[0]
            pre_norm=[pr/presum for pr in pre[1:]]
            pre_clas=pre_norm.index(max(pre_norm))
            pre_P=max(pre_norm)
            wsi_id=line.split('_files_')[0].split('test_')[1]
            if pre_clas!=2:
                if wsi_id in wsi_result:
                    if pre_clas in wsi_result[wsi_id]:
                        wsi_result[wsi_id][pre_clas][0]+=pre_P
                        wsi_result[wsi_id][pre_clas][1]+=1
                    else:
                        wsi_result[wsi_id][pre_clas]=[]
                        wsi_result[wsi_id][pre_clas].append(pre_P)
                        wsi_result[wsi_id][pre_clas].append(1)
                else:
                    wsi_result[wsi_id]={}
                    wsi_result[wsi_id][pre_clas]=[]
                    wsi_result[wsi_id][pre_clas].append(pre_P)
                    wsi_result[wsi_id][pre_clas].append(1)

    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('predominant_minor')
    worksheet.write(0,0,label='WSI id')
    worksheet.write(0,1,label='predict dominant:class num P')
    worksheet.write(0,2,label='predict minor:class num P')
    for i in range(len(wsi_result.keys())):
        predict=list(wsi_result.values())[i]
        pre_num=[predict[c][1] for c in predict]
        #print(predict)
        dominant=sorted(pre_num)[-1]
        minor=sorted(pre_num)[-2]

        dominant_index=list(predict.keys())[pre_num.index(dominant)]
        minor_index=list(predict.keys())[pre_num.index(minor)]
        dominant_P=round(predict[dominant_index][0]/dominant,2)
        minor_P=round(predict[minor_index][0]/minor,2)

        worksheet.write(i+1,0,label=list(wsi_result.keys())[i])
        worksheet.write(i+1,1,label=clas[dominant_index]+'_'+str(dominant)+'_'+str(dominant_P))
        worksheet.write(i+1,2,label=clas[minor_index]+'_'+str(minor)+'_'+str(minor_P))
    workbook.save(save_path)
def dominant_minor_clas_all_p(test_path,save_path):
    clas=['AAH','AIS','BG','LPA','MIA']
    with open(test_result,'r') as f:
        wsi_result={}
        #wsi_count={}
        for line in f:
            prestr=line.split('[')[1].split(']')[0].split()
            pre=[float(pr) for pr in prestr]
            presum=1-pre[0]
            pre_norm=[pr/presum for pr in pre[1:]]
            pre_clas=pre_norm.index(max(pre_norm))
            pre_P=max(pre_norm)
            wsi_id=line.split('_files_')[0].split('test_')[1]
            if pre_clas!=2:
                if wsi_id in wsi_result:
                    if pre_clas in wsi_result[wsi_id]:
                        wsi_result[wsi_id][pre_clas][0]+=pre_P
                        wsi_result[wsi_id][pre_clas][1]+=1
                    else:
                        wsi_result[wsi_id][pre_clas]=[]
                        wsi_result[wsi_id][pre_clas].append(pre_P)
                        wsi_result[wsi_id][pre_clas].append(1)
                else:
                    wsi_result[wsi_id]={}
                    wsi_result[wsi_id][pre_clas]=[]
                    wsi_result[wsi_id][pre_clas].append(pre_P)
                    wsi_result[wsi_id][pre_clas].append(1)

    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('predominant_minor')
    worksheet.write(0,0,label='WSI id')
    worksheet.write(0,1,label='predict dominant:class num P')
    worksheet.write(0,2,label='predict minor:class num P')
    for i in range(len(wsi_result.keys())):
        predict=list(wsi_result.values())[i]
        pre_p=[predict[c][0] for c in predict]
        #print(predict)
        dominant=sorted(pre_p)[-1]
        minor=sorted(pre_p)[-2]
        
        dominant_index=list(predict.keys())[pre_p.index(dominant)]
        minor_index=list(predict.keys())[pre_p.index(minor)]
        dominant_p=round(predict[dominant_index][0],2)
        minor_p=round(predict[minor_index][0],2)

        worksheet.write(i+1,0,label=list(wsi_result.keys())[i])
        worksheet.write(i+1,1,label=clas[dominant_index]+'_'+str(dominant_p))
        worksheet.write(i+1,2,label=clas[minor_index]+'_'+str(minor_p))
    workbook.save(save_path)
def dominant_minor_clas_ave_p(test_path,save_path):
    clas=['AAH','AIS','BG','LPA','MIA']
    with open(test_result,'r') as f:
        wsi_result={}
        #wsi_count={}
        for line in f:
            prestr=line.split('[')[1].split(']')[0].split()
            pre=[float(pr) for pr in prestr]
            presum=1-pre[0]
            pre_norm=[pr/presum for pr in pre[1:]]
            pre_clas=pre_norm.index(max(pre_norm))
            pre_P=max(pre_norm)
            wsi_id=line.split('_files_')[0].split('test_')[1]
            if pre_clas!=2:
                if wsi_id in wsi_result:
                    if pre_clas in wsi_result[wsi_id]:
                        wsi_result[wsi_id][pre_clas][0]+=pre_P
                        wsi_result[wsi_id][pre_clas][1]+=1
                    else:
                        wsi_result[wsi_id][pre_clas]=[]
                        wsi_result[wsi_id][pre_clas].append(pre_P)
                        wsi_result[wsi_id][pre_clas].append(1)
                else:
                    wsi_result[wsi_id]={}
                    wsi_result[wsi_id][pre_clas]=[]
                    wsi_result[wsi_id][pre_clas].append(pre_P)
                    wsi_result[wsi_id][pre_clas].append(1)

    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('predominant_minor')
    worksheet.write(0,0,label='WSI id')
    worksheet.write(0,1,label='dominant ave_P')
    worksheet.write(0,2,label='minor ave_P')
    for i in range(len(wsi_result.keys())):
        predict=list(wsi_result.values())[i]
        pre_ave_p=[predict[c][0]/predict[c][1] for c in predict]
        #print(predict)
        dominant=sorted(pre_ave_p)[-1]
        minor=sorted(pre_ave_p)[-2]
        
        dominant_index=list(predict.keys())[pre_ave_p.index(dominant)]
        minor_index=list(predict.keys())[pre_ave_p.index(minor)]

        worksheet.write(i+1,0,label=list(wsi_result.keys())[i])
        worksheet.write(i+1,1,label=clas[dominant_index]+'_'+str(round(dominant,2)))
        worksheet.write(i+1,2,label=clas[minor_index]+'_'+str(round(minor,2)))
    workbook.save(save_path)
def test_acc(testpath):
    for i in os.listdir(testpath):
        pass

if __name__ == "__main__":
    '''daping'''
    # test_result='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/train_test/test_960k/out_filename_Stats.txt'
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/confusion_Train_num.png'
    # confusion_mx(test_result,save_path,'Train')
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/confusion_Train_%.png'
    # confusion_mx_percent(test_result,save_path,'Train')

    # test_result='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result_test_valid_TFrecord/test/test_960k/out_filename_Stats.txt'
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/confusion_valid_num.png'
    # confusion_mx(test_result,save_path,'Valid')
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/confusion_valid_%.png'
    # confusion_mx_percent(test_result,save_path,'Valid')

    # test_result='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/test_test/test_960k/out_filename_Stats.txt'
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/confusion_test_num.png'
    # confusion_mx(test_result,save_path,'Test')
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/confusion_test_%.png'
    # confusion_mx_percent(test_result,save_path,'Test')
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/confusion_test_%_noBG.png'
    # confusion_mx_percent_noBG(test_result,save_path,'Test')

    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/predominant_minor_class_all_p.xls'
    # dominant_minor_clas_all_p(test_result,save_path)
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result_test_overlap/predominant_minor_class_all_p.xls'
    # dominant_minor_clas_all_p(test_result,save_path)
    # save_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result_test_overlap/predominant_minor_class_num.xls'
    # dominant_minor_clas_num(test_result,save_path)

    # test_result='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/train_test/test_960k/out_filename_Stats.txt'
    # savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/train_acc_recal_f1.xls'
    # accuracy_revall(test_result,savepath)
    
    # test_result='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result_test_valid_TFrecord/test/test_960k/out_filename_Stats.txt'
    # savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/valid_acc_recal_f1.xls'
    # accuracy_revall(test_result,savepath)

    # test_result='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/test_test/test_960k/out_filename_Stats.txt'
    # savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/test_acc_recal_f1.xls'
    # accuracy_revall(test_result,savepath)

    test_result='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/test_test/test_960k/out_filename_Stats.txt'
    savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/test_hist_'
    #bar(test_result,savepath)
    #hist(test_result,savepath)

    test_result='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/test_test/test_960k/out_filename_Stats.txt'
    savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/LPA_test_class'
    #symlink_AIS(test_result,savepath)
    '''xinjiang'''
    # test_result='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/train_test/test_4600k/out_filename_Stats.txt'
    # save_path='//disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/pic_of_confusion_else/confusion_Train_num.png'
    # confusion_mx(test_result,save_path,'Train')
    # save_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/pic_of_confusion_else/confusion_Train_%.png'
    # confusion_mx_percent(test_result,save_path,'Train')

    # test_result='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/valid_test/test_4600k/out_filename_Stats.txt'
    # save_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/pic_of_confusion_else/confusion_valid_num.png'
    # confusion_mx(test_result,save_path,'Valid')
    # save_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/pic_of_confusion_else/confusion_valid_%.png'
    # confusion_mx_percent(test_result,save_path,'Valid')

    # test_result='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/test/test_4600k/out_filename_Stats.txt'
    # save_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/pic_of_confusion_else/confusion_test_num.png'
    # confusion_mx(test_result,save_path,'Test')
    # save_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/pic_of_confusion_else/confusion_test_%.png'
    # confusion_mx_percent(test_result,save_path,'Test')

    '''paper_result_table of single model'''
    test_result='/disk1/zhangyingxin/project/lung/daping_grade2/paper_script/test/result/0/test_700k/out_filename_Stats.txt'
    savepath='/disk1/zhangyingxin/project/lung/daping_grade2/paper_script/table_pic_of_paper/paper_acc_recal_f1.xls'
    model=''
    #paper_result_table(test_result,savepath,model)
    '''paper_result_table of multi model 
    可同时得到多个模型结果的统计
    root='/disk1/zhangyingxin/project/lung/daping_grade2/paper_script/test/result'
    result_list=[]
    for t_id in os.listdir(root):
        for fi in os.listdir(os.path.join(root,t_id)):
            if 'test' in fi:
                root_=os.path.join(root,t_id,fi)
                result_list.append(os.path.join(root_,'out_filename_Stats.txt'))
    baggingpath=[root]
    savepath='/disk1/zhangyingxin/project/lung/daping_grade2/paper_script/table_pic_of_paper/bagging.xls'
    paper(result_list,savepath,baggingpath)'''
    '''paper_result_table of multi model 
    可同时得到多个模型结果的统计  xinjiang paper'''
    root='/disk2/zhangyingxin/xinjiang_paper_supply/20X/bagging'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level_add3rdata'
    result_list=[]
    model_list=['result_nopreweight_1000step']#'result','result_nopreweight','result_tune_istrainingF'
    for t_id in os.listdir(root):
        for m in os.listdir(os.path.join(root,t_id)):
            if m in model_list:
            #if 'result' in m:
                root_=os.path.join(root,t_id,m,'test')
                for fi in os.listdir(root_):
                    if 'test' in fi:
                        rot_=os.path.join(root_,fi)
                        result_list.append(os.path.join(rot_,'out_filename_Stats.txt'))
    baggingpath=[root]
    savepath='/disk2/zhangyingxin/xinjiang_paper_supply/20X/summary/test_result_nopreweight_1000step.xls'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary/test_result_istrainingF_add3rdata.xls'
    
    #paper_xinjiang(result_list,savepath,model_list,baggingpath)
    '''xinjaing paper bagging 9st exam confusion matrix
    root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment_v1/train_valid_test/bagging_deepath'
    #
    out='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary/is_trainingF/confusion_test_deepath_bagging'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary/confusion_test_deepath_bagging'
    for t_id in os.listdir(root):
        for f in os.listdir(os.path.join(root,str(t_id),'result_tune_istrainingF','test')):
            if 'test' in f:
                test_result=os.path.join(root,str(t_id),'result_tune_istrainingF','test',f,'out_filename_Stats.txt')
                # save_path=out+str(t_id)+'_num.png'
                # confusion_mx(test_result,save_path,'Test')
                save_path=out+str(t_id)+'_%.png'
                print(save_path)
                confusion_mx_percent(test_result,save_path,str(t_id)+' Test')'''
    '''get bagging roc value txt in class
    baggingpath[0]:  root path of deepath/deepslide model test result
    model_list[0]:  name of certain model 
    '''
    output='/disk2/zhangyingxin/xinjiang_paper_supply/20X/summary/deepath'
    for i in range(len(model_list)):
        output_dir=output+'_'+model_list[i]+'_bagging_roc_txt'
        paper_bagging_xinjiang(baggingpath[0],model_list[i],savepath='',model='',output_dir=output_dir)
    '''计算两个模型混合的bagging 结果'''
    bagging_istrainingF=[0,3,4,5,7,8,9]
    bagging_3rdata=[1,2,6]
    root_istrainingF='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level'
    root_3rdata='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level_add3rdata'
    test_list=[]
    for i in bagging_istrainingF:
        test_list.append(os.path.join(root_istrainingF,str(i)))
    for i in bagging_3rdata:
        test_list.append(os.path.join(root_3rdata,str(i)))
    # print(test_list)
    # sys.exit(0)
    model_name=['result_tune_istrainingF','result']
    output='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary/deepath'
    output_dir=output+'_mix_bagging_roc_txt'
    #paper_bagging_xinjiang_2modelmix(test_list,model_name,savepath='',model='',output_dir='')
    '''calculate kappa score according cofusion_matrix'''
    test_result='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level/0/result/test/test_840k/out_filename_Stats.txt'
    root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level'
    for t_id in os.listdir(root):
        for f in os.listdir(os.path.join(root,str(t_id),'result_tune_istrainingF','test')):
            if 'test' in f:
                test_result=os.path.join(root,str(t_id),'result_tune_istrainingF','test',f,'out_filename_Stats.txt')
                #acording_confusionmx_get_kappa(test_result)
    '''confusion of xj last exam'''
    test_result='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/workspace_1/result/test/test_13392k/out_filename_Stats.txt'
    save_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/512workspace_maxiaomei_limit_normalization/confusion_test_num.png'
    #confusion_mx(test_result,save_path,'Test')
    save_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/workspace_1/confusion_test_%_manydoc.png'
    #confusion_mx_percent(test_result,save_path,'Test')


