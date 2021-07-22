import os
import csv
modelpath='/disk1/zhangyingxin/project/lung/deepslide-master/checkpoints'
result='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_deepslide_alex'
for i in os.listdir(result):
    if i!='0':
        continue
    result_=os.path.join(result,i)
    csv_r=os.listdir(result_)
    for c in csv_r:
        if c.endswith('.csv'):
            csv_=c
    out=os.path.join(result_,'checkpoints')
    if not os.path.exists(out):
        os.makedirs(out)
    
    with open(os.path.join(result_,csv_),'r') as f:
        reader=csv.reader(f)
        val_acc=[]
        for row in reader:
            if row[0]!='epoch':
                val_acc.append(row[0]+'_'+str(round(float(row[4]),4)))
        
        for m in os.listdir(modelpath):
            vacc=m.split('_')[1].split('e')[1]+'_'+str(round(float(os.path.splitext(m)[0].split('va')[1]),4))
            if vacc in val_acc:
                fro=os.path.join(modelpath,m)
                to=os.path.join(out,m)
                os.symlink(fro,to)
