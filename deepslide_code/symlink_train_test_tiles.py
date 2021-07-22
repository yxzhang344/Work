import os
def symlink(from_path,to_path,mag):
    types=os.listdir(from_path)
    for typ in types:
        if typ in ['train','valid']:
            if typ=='valid':
                save=os.path.join(to_path,'val')
            else:
                save=os.path.join(to_path,typ)
            clas=os.listdir(os.path.join(from_path,typ))
            for cla in clas:
                save_to=os.path.join(save,cla)
                if not os.path.exists(save_to):
                    os.makedirs(save_to)
                files=os.listdir(os.path.join(from_path,typ,cla))
                for fil in files:
                    fil_id=fil.split('files')[0]
                    tiles_path=os.path.join(from_path,typ,cla,fil,mag)
                    for tile in os.listdir(tiles_path):
                        save_path=os.path.join(save_to,fil_id+tile)#.split('.')[0]+'.jpeg'
                        os.symlink(os.path.join(tiles_path,tile),save_path)
def symlink_test(from_path,to_path,mag):
    save=to_path
    clas=os.listdir(from_path)
    for cla in clas:
        save_to=os.path.join(save,cla,cla)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        files=os.listdir(os.path.join(from_path,cla))
        for fil in files:
            if not fil.endswith('.dzi'):
                fil_id=fil.split('files')[0]
                tiles_path=os.path.join(from_path,cla,fil,mag)
                for tile in os.listdir(tiles_path):
                    save_path=os.path.join(save_to,fil_id+tile)#.split('.')[0]+'.jpeg'
                    os.symlink(os.path.join(tiles_path,tile),save_path)
if __name__ == "__main__":
    '''daping'''
    from_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1/512px_tiled'
    to_path='/disk1/zhangyingxin/project/lung/deepslide-master/train_folder'
    mag='5.0'
    #symlink(from_path,to_path,mag)
    '''xinjiang paper
    from_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_train_valid_bagging_patient_level_20Xfro5X'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment_v1/512px_Tiled_train_valid_bagging_patient_enhance/'
    to_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_train_valid_bagging_patient_level_20Xfro5X_deepslide'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment_v1/512px_Tiled_train_valid_bagging_patient_enhance_deepslide'
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    mag='20.0'
    for i in range(10):
        fro=os.path.join(from_path,str(i))
        to=os.path.join(to_path,str(i))
        symlink(fro,to,mag)'''
    '''xinjiang paper test'''
    from_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_test_20X'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_test'
    to_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_test_20X_deepslide'
    mag='20.0'
    symlink_test(from_path,to_path,mag)