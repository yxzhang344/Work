import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
import csv
def loss_line(lossfile,savepath):
    with open(lossfile,'r') as f:
        losslist=[]
        for line in f:
            loss=float(line.split('loss = ')[1].split(' (')[0])
            losslist.append(loss)
    print(losslist)
    x=list(range(len(losslist)))
    plt.figure()
    plt.plot(x,losslist)
    plt.title('Train Loss Line')
    #plt.legend()
    plt.xlabel('Steps*10')
    plt.ylabel('Loss')
    plt.savefig(savepath)
def my_loss_line(trainloss,validloss,testloss,savepath):
    with open(trainloss,'r') as f:
        losslist_t=[]
        for line in f:
            loss=float(line.split('Loss:')[1].strip('\n'))
            losslist_t.append(loss)
    with open(validloss,'r') as f:
        losslist_v=[]
        for line in f:
            loss=float(line.split('Loss:')[1].strip('\n'))
            losslist_v.append(loss)
    with open(testloss,'r') as f:
        losslist_te=[]
        for line in f:
            loss=float(line.split('Loss:')[1].strip('\n'))
            losslist_te.append(loss)
    #print(losslist)
    x=list(range(len(losslist_t)))
    plt.figure()
    plt.plot(x,losslist_t,label='train')
    plt.plot(x,losslist_v,label='valid')
    plt.plot(x[:-1],losslist_te,label='test')
    plt.title('Train Valid and Test Loss Line')
    plt.legend()
    plt.xlabel('Steps*120')
    plt.ylabel('Loss')
    plt.savefig(savepath)
def ROC_line(root_roc,roc_bagging_list,subexam_root,subexam_name,savepath):
    '''
    root_roc  各种.._roc_txt存放的上级目录
    roc_bagging_list  各种.._roc_txt组成的列表
    subexam_root  子实验目录
    subexam_name  子实验名称
    '''
    sub=['Acinar','Normal','LPA','Micropapillary','Papillary','Solid']
    classes=['c1auc','c2auc','c3auc','c4auc','c5auc','c6auc']
    subplt=[231,232,233,234,235,236]
    plt.cla()
    fig=plt.figure(figsize=(9,6))
    plt.title('ROC Curve')
    for c in classes:
        dp=0
        plt.subplot(subplt[classes.index(c)])
        for r in roc_bagging_list:
            roc_path=os.path.join(root_roc,r)
            for roc in os.listdir(roc_path):
                if roc.split('_')[4]==c:
                    x=[]
                    y=[]
                    #auc=round(float(roc.split('_')[-1].strip('.txt')),2)
                    auc=str(round(float(roc.split('_')[-1].strip('.txt')),2))
                    with open(os.path.join(roc_path,roc),'r') as f:
                        for l in f:
                            x.append(float(l.split('\t')[0]))
                            y.append(float(l.split('\t')[1].split('\n')[0]))
                    if r.split('_')[0]=='deepath':
                        if 'nopreweight' in r:
                            plt.plot(x,y,label=auc+'In_Bag')
                        else:
                            dp+=1
                            plt.plot(x,y,label=auc+'In_Bag_Pre')#+str(dp)
                    else:
                        if 'preweight' in r:
                            plt.plot(x,y,label=auc+'Re_Bag_Pre')
                        else:
                            dp+=1
                            plt.plot(x,y,label=auc+'Re_Bag')
                        #plt.plot(x,y,label='DeepSlide')
                    #print(x[:10],y[:10])
                    break
        # for idx in os.listdir(subexam_root):
        #     testpath=os.path.join(subexam_root,idx,subexam_name,'test')
        #     for f_ in os.listdir(testpath):
        #         if f_.endswith('k'):
        #             for roc in os.listdir(os.path.join(testpath,f_)):
        #                 #print(roc)
        #                 if 'AvPb' in roc:
        #                     if roc.split('_')[4]==c:
        #                         x=[]
        #                         y=[]
        #                         with open(os.path.join(testpath,f_,roc),'r') as f:
        #                             for l in f:
        #                                 x.append(float(l.split('\t')[0]))
        #                                 y.append(float(l.split('\t')[1].split('\n')[0]))
        #                         plt.plot(x,y,label=subexam_name+idx)
        #                         break
        #             break
        plt.title(sub[classes.index(c)])#+'_ROC_curve'
        plt.legend()
        plt.xlabel('fpr')
        plt.ylabel('tpr')
    fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
    plt.savefig(os.path.join(savepath,'ROC_curve_bagging_paper_auc_v1.png'))
        #plt.savefig(os.path.join(savepath,'ROC_curve_bagging_'+sub[classes.index(c)]+'.png'))
        #plt.close(0)
        #print('finish',sub[classes.index(c)])

                
def train_valid_acc_line(trainacc,validacc,testacc,savepath):
    with open(trainacc,'r') as f:
        acclist_t=[]
        for line in f:
            print(line)
            acc=round(float(line.split('Precision:')[1].strip('\n')),2)
            acclist_t.append(acc)
    with open(validacc,'r') as f:
        acclist_v=[]
        for line in f:
            print(line)
            acc=round(float(line.split('Precision:')[1].strip('\n')),2)
            acclist_v.append(acc)
    with open(testacc,'r') as f:
        acclist_te=[]
        for line in f:
            print(line)
            acc=round(float(line.split('Precision:')[1].strip('\n')),2)
            acclist_te.append(acc)
    #print(losslist)
    x=list(range(len(acclist_te)))
    plt.figure()
    plt.plot(x[:-1],acclist_t,label='train')
    
    plt.plot(x,acclist_v,label='valid')
    plt.plot(x,acclist_te,label='test')
    plt.title('Train Valid and Test Accuracy Line')
    plt.legend()
    plt.xlabel('Steps*2300')
    plt.ylabel('Accuracy')
    plt.savefig(savepath)
def train_valid_acc_line_bagging_deepath(trainacc_list,validacc_list,valid_step,savepath):
    acclist_ts=[]
    acclist_vs=[]
    for trainacc in trainacc_list:
        with open(trainacc,'r') as f:
            acclist_t=[]
            for line in f:
                print(line)
                acc=round(float(line.split('acc = ')[1].split(' (')[0]),2)
                acclist_t.append(acc)
            acclist_ts.append(acclist_t)
    for validacc in validacc_list:
        with open(validacc,'r') as f:
            acclist_v=[]
            for line in f:
                print(line)
                acc=round(float(line.split('Precision:')[1].strip('\n')),2)
                acclist_v.append(acc)
            acclist_vs.append(acclist_v)
    #print(losslist)
    tx_list=[]
    vx_list=[]
    
    for i in range(len(acclist_ts)):
        tx_list.append(list(np.array(list(range(len(acclist_ts[i]))))*10))
        vx_list.append(list(np.array(list(range(len(acclist_vs[i]))))*valid_step[i]))
    plt.figure()
    
    #sub=[521,522,523,524,525,526,527,528,529]
    sub=[231,232,233,234,235,236]
    for i in range(6):
        #print(tx_list[i],vx_list[i])
        plt.subplot(sub[i])
        plt.plot(tx_list[i],acclist_ts[i],label=str(i)+'_train')
        plt.plot(vx_list[i],acclist_vs[i],label=str(i)+'_valid')
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        if i==1:
            plt.title('Deepath Bagging Train Valid Accuracy Line')
        
    # plt.subplot(5,2,10)
    # plt.plot(tx_list[9],acclist_ts[9],label=str(9)+'_train')
    # plt.plot(vx_list[9],acclist_vs[9],label=str(9)+'_valid')
    # plt.legend()
    
    plt.savefig(savepath)
def train_valid_acc_line_bagging_deepslide(result_list,savepath):
    acclist_ts=[]
    acclist_vs=[]
    for res in result_list:
        with open(res,'r') as f:
            reader=csv.reader(f)
            acclist_t=[]
            acclist_v=[]
            for line in reader:
                if line[0]!='epoch':
                    acc=round(float(line[2]),2)
                    acclist_t.append(acc)
                    acc=round(float(line[4]),2)
                    acclist_v.append(acc)
            acclist_ts.append(acclist_t)
            acclist_vs.append(acclist_v)
    #print(losslist)
    x_list=[]
    
    for i in range(len(acclist_ts)):
        x_list.append(list(range(len(acclist_ts[i]))))
    plt.figure()
    
    #sub=[521,522,523,524,525,526,527,528,529]
    sub=[231,232,233,234,235,236]
    for i in range(6):
        #print(tx_list[i],vx_list[i])
        plt.subplot(sub[i])
        plt.plot(x_list[i],acclist_ts[i],label=str(i)+'_train')
        plt.plot(x_list[i],acclist_vs[i],label=str(i)+'_valid')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        if i==1:
            plt.title('Deepslide Bagging Train Valid Accuracy Line')
        
    # plt.subplot(5,2,10)
    # plt.plot(tx_list[9],acclist_ts[9],label=str(9)+'_train')
    # plt.plot(vx_list[9],acclist_vs[9],label=str(9)+'_valid')
    # plt.legend()
    
    plt.savefig(savepath)
def acc_line(accfile,savepath):
    with open(accfile,'r') as f:
        acclist=[]
        for line in f:
            print(line)
            acc=round(float(line.split('Precision:')[1].strip('\n')),2)
            acclist.append(acc)
    #print(losslist)
    x=list(range(len(acclist)))
    plt.figure()
    plt.plot(x,acclist)
    plt.title('Valid Accuracy Line')
    #plt.legend()
    plt.xlabel('Steps*120')
    plt.ylabel('Accuracy')
    plt.savefig(savepath)
if __name__ == "__main__":
    '''daping'''
    # savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result_15epoch_decay_5X/trainloss.png'
    # lossfile='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result_15epoch_decay_5X/train/training_loss.txt'
    # loss_line(lossfile,savepath)
    # accfile='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result_15epoch_decay_5X/valid/precision_at_1.txt'
    # savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result_15epoch_decay_5X/validacc.png'
    # acc_line(accfile,savepath)

    # trainloss='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/train_loss/loss.txt'
    # testloss='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/test_loss/loss.txt'
    # validloss='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/valid_loss/loss.txt'
    # savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/train_valid_loss.png'
    # my_loss_line(trainloss,validloss,testloss,savepath)

    # trainacc='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/train_loss/precision_at_1.txt'
    # validacc='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/valid_loss/precision_at_1.txt'
    # testacc='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/test_loss/precision_at_1.txt'
    # savepath='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/result/train_valid_acc.png'
    # train_valid_acc_line(trainacc,validacc,testacc,savepath)
    '''xinjiang'''
    trainacc='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/train_loss/precision_at_1.txt'
    validacc='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/valid_loss/precision_at_1.txt'
    testacc='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/test_loss/precision_at_1.txt'
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/5X_workspace/result/pic_of_confusion_else/train_valid_acc.png'
    #train_valid_acc_line(trainacc,validacc,testacc,savepath)

    '''xinjiang grade2 bagging deepath
    root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level'
    trainacc_list=[]
    validacc_list=[]
    valid_step=[]
    for i in range(10):
        CHECKPOINT_PATH=os.path.join(root,str(i),'result','train')
        model=os.listdir(CHECKPOINT_PATH)
        idex=[]
        for m in model:
            if m.endswith('.meta'):
                idex.append(int(os.path.splitext(m)[0].split('-')[1]))
        m_id=sorted(idex)
        valid_step.append(m_id[1])
    for i in sorted(os.listdir(root)):
        trainacc_list.append(os.path.join(root,str(i),'result','train','training_loss_acc.txt'))
        validacc_list.append(os.path.join(root,str(i),'result','valid','precision_at_1.txt'))
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary/bagging_train_valid_acc_deepath.png'
    train_valid_acc_line_bagging_deepath(trainacc_list,validacc_list,valid_step,savepath)
    '''
    '''xinjiang grade2 bagging deepslide'''
    root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_deepslide_patient_level'
    result_list=[]
    for i in sorted(os.listdir(root)):
        for f in os.listdir(os.path.join(root,str(i))):
            if f.endswith('.csv'):
                result_list.append(os.path.join(root,str(i),f))
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary/bagging_train_valid_acc_deepslide.png'
    #train_valid_acc_line_bagging_deepslide(result_list,savepath)

    '''roc curve'''
    root_roc='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary'
    roc_bagging_list=['deepath_result_nopreweight_bagging_roc_txt','deepath_result_bagging_roc_txt',
     'deepslide_bagging_roc_txt','deepslide_preweight_mopdel16_bagging_roc_txt']
    #'deepath_result_tune_istrainingF_bagging_roc_txt',,'deepath_result_tune_istrainingF_6out_bagging_roc_txt'
    
    subexam_root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level'
    subexam_name='result'
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary/ROC_curve'
    ROC_line(root_roc,roc_bagging_list,subexam_root,subexam_name,savepath)