import os
import xlwt
import xlrd
import numpy as np
def count_all_tiles(tile_root,savepath):
    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('xummary')
    worksheet.write(0,0,label='class')
    worksheet.write(0,1,label='svs')
    worksheet.write(0,2,label='tile')
    i = 0
    #cla=['BG_big']
    #xinjaing:  
    cla=['BG','Acinar','LPA','Micropapillary','Papillary','Solid']#_noverlap
    '''preweigt'''
    #cla=['luad','lusc','normal']
    #cla=['AAH','AIS','LPA','MIA','BG']#daping
    #blank=dict(AAH=[],AIS=[],LPA=[],MIA=[],BG=[])
    mag='5.0'
    blank=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])#
    #blank=dict(luad=[],lusc=[],normal=[])
    #"/mnt/share/zhangyingxin/lung_local/xinjiang_2grade/paper_exam_try/doctor_tiles/liuting_group3/512px_Tiled/"         xinjiang
    clas=os.listdir(tile_root)
    for cl in clas:
        if cl in cla:
            i += 1
            j=num=0
            worksheet.write(i,0,label=cl)
            files=os.listdir(os.path.join(tile_root,cl))
            files_n=[]
            for fil in files:
                if os.path.splitext(fil)[1]=='':
                #if os.path.splitext(os.path.splitext(fil)[1])[1]=='':  using for file has '.'
                    if os.path.exists(os.path.join(tile_root,cl,fil,mag)) and os.listdir(os.path.join(tile_root,cl,fil,mag))!=[]:
                        #print(cl,fil,len(os.listdir(os.path.join(tile_root,cl,fil,'5.0'))))
                        num += len(os.listdir(os.path.join(tile_root,cl,fil,mag)))
                        #print(num)
                        j += 1
                        files_n.append(fil+'_'+str(len(os.listdir(os.path.join(tile_root,cl,fil,mag)))))
                    else:
                        #print(cl)
                        blank[cl].append(fil)
            worksheet.write(i,1,label=str(j))
            worksheet.write(i,2,label=str(num))
            worksheet.write(i,3,label=str(files_n))
    workbook.save(savepath)
    print(blank)

def count_all_tiles_patient_level(tile_root,summaryf,savepath):
    book=xlrd.open_workbook(summaryf)
    sheet=book.sheet_by_name("test_patients")
    test_id=sheet.col_values(0)

    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('xummary')
    worksheet.write(0,0,label='class')
    worksheet.write(0,1,label='patient_n')
    worksheet.write(0,2,label='tile')
    i = 0
    #cla=['BG_big']
    #xinjaing:  
    cla=['BG','Acinar','LPA','Micropapillary','Papillary','Solid']#
    #cla=['AAH','AIS','LPA','MIA','BG']#daping
    #blank=dict(AAH=[],AIS=[],LPA=[],MIA=[],BG=[])
    mag='5.0'
    blank=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])#
    #"/mnt/share/zhangyingxin/lung_local/xinjiang_2grade/paper_exam_try/doctor_tiles/liuting_group3/512px_Tiled/"         xinjiang
    clas=os.listdir(tile_root)
    for cl in clas:
        if cl in cla:
            i += 1
            j=num=0
            worksheet.write(i,0,label=cl)
            files=os.listdir(os.path.join(tile_root,cl))
            #files_n={}
            # fil_nums=[]
            # for fil in files:
            #     if os.path.splitext(fil)[1]=='':
            #         pat_id='-'.join(fil.split('-')[:2])
            #         if not (pat_id in test_id) and os.path.exists(os.path.join(tile_root,cl,fil,mag)) and os.listdir(os.path.join(tile_root,cl,fil,mag))!=[]:
            #             #print(cl,fil,len(os.listdir(os.path.join(tile_root,cl,fil,'5.0'))))
            #             ntile=len(os.listdir(os.path.join(tile_root,cl,fil,mag)))
            #             num += ntile
            #             j += 1
            #             fil_nums.append(fil+'_'+str(ntile))
            #         else:
            #             #print(cl)
            #             blank[cl].append(fil)
            #均衡tiles 数量看患者界别tiles时用
            files_n={}
            for fil in files:
                if os.path.splitext(fil)[1]=='':
                    pat_id='-'.join(fil.split('-')[:2])
                    if not (pat_id in test_id) and os.path.exists(os.path.join(tile_root,cl,fil,mag)) and os.listdir(os.path.join(tile_root,cl,fil,mag))!=[]:
                        # 
                        #print(cl,fil,len(os.listdir(os.path.join(tile_root,cl,fil,'5.0'))))
                        ntile=len(os.listdir(os.path.join(tile_root,cl,fil,mag)))
                        num += ntile
                        
                        if pat_id in files_n:
                            files_n[pat_id]+=ntile
                        else:
                            j += 1
                            files_n[pat_id]=ntile
                    else:
                        #print(cl)
                        blank[cl].append(fil)
            worksheet.write(i,1,label=str(j))
            worksheet.write(i,2,label=str(num))
            worksheet.write(i,3,label=str(files_n))#fil_nums
    workbook.save(savepath)
    print(blank)

def count_train_valid_tiles(tile_root,out_path,mag):
    workbook=xlwt.Workbook()
    typ=os.listdir(tile_root)
    for ty in typ:
        blank=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
        #blank=dict(AIS=[],AAH=[],LPA=[],MIA=[],BG=[])
        worksheet=workbook.add_sheet('summary_'+ty)
        worksheet.write(0,0,label='class')
        worksheet.write(0,1,label='svs')
        worksheet.write(0,2,label='tile')
        i = 0
        clas=os.listdir(os.path.join(tile_root,ty))
        for cl in clas:
            i += 1
            j=num=0
            worksheet.write(i,0,label=cl)
            files=os.listdir(os.path.join(tile_root,ty,cl))
            fil_nums=[]
            for fil in files:
                if os.path.splitext(fil)[1]=='':
                    if os.listdir(os.path.join(tile_root,ty,cl,fil,mag))!=[]:
                        #print(cl,fil,len(os.listdir(os.path.join(tile_root,cl,fil,'5.0'))))
                        num += len(os.listdir(os.path.join(tile_root,ty,cl,fil,mag)))
                        j += 1
                        fil_nums.append(fil+'_'+str(len(os.listdir(os.path.join(tile_root,ty,cl,fil,mag)))))
                    else:
                        #print(cl)
                        blank[cl].append(fil)
            worksheet.write(i,1,j)
            worksheet.write(i,2,num)
            worksheet.write(i,3,str(fil_nums))

        print(ty,' bank is ',blank)
    workbook.save(out_path)
def count_train_valid_tiles_bagging(tile_root,out_path,mag,n_exam):
    workbook=xlwt.Workbook()
    
    for n in range(n_exam):
        print(os.path.join(tile_root,str(n)),os.path.exists(os.path.join(tile_root,str(n))))
        typ=os.listdir(os.path.join(tile_root,str(n)))
        for ty in typ:
            blank=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
            #blank=dict(AIS=[],AAH=[],LPA=[],MIA=[],BG=[])
            worksheet=workbook.add_sheet(str(n)+'_'+ty)
            worksheet.write(0,0,label='class')
            worksheet.write(0,1,label='svs')
            worksheet.write(0,2,label='tile')
            i = 0
            clas=os.listdir(os.path.join(tile_root,str(n),ty))
            for cl in clas:
                i += 1
                j=num=0
                worksheet.write(i,0,label=cl)
                files=os.listdir(os.path.join(tile_root,str(n),ty,cl))
                fil_nums=[]
                for fil in files:
                    if os.path.splitext(fil)[1]=='':
                        if os.listdir(os.path.join(tile_root,str(n),ty,cl,fil,mag))!=[]:
                            #print(cl,fil,len(os.listdir(os.path.join(tile_root,cl,fil,'5.0'))))
                            num += len(os.listdir(os.path.join(tile_root,str(n),ty,cl,fil,mag)))
                            j += 1
                            fil_nums.append(fil+'_'+str(len(os.listdir(os.path.join(tile_root,str(n),ty,cl,fil,mag)))))
                        else:
                            #print(cl)
                            blank[cl].append(fil)
                worksheet.write(i,1,j)
                worksheet.write(i,2,num)
                worksheet.write(i,3,str(fil_nums))

            print(ty,' bank is ',blank)
    workbook.save(out_path)
def count_mean_std(xlspath,savepath):
    # cla=['BG','AAH','AIS','LPA','MIA']#
    # out=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # out_v=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # svs=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # tile=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # svs_v=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # tile_v=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    cla=['BG','Acinar','LPA','Micropapillary','Papillary','Solid']#
    out=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    out_v=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    svs=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    tile=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    svs_v=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    tile_v=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    for ff in os.listdir(xlspath):
        if ff.endswith('.xls'):
            ff_=os.path.join(xlspath,ff)
            book=xlrd.open_workbook(ff_)
            sheet_train=book.sheet_by_name("xummary_train")
            sheet_valid=book.sheet_by_name("xummary_valid")
            for i in range(5):
                row=sheet_train.row_values(i+1)
                cla=row[0]
                n_svs=int(row[1])
                n_tile=int(row[2])
                svs[cla].append(n_svs)
                tile[cla].append(n_tile)

                row=sheet_valid.row_values(i+1)
                cla=row[0]
                n_svs=int(row[1])
                n_tile=int(row[2])
                svs_v[cla].append(n_svs)
                tile_v[cla].append(n_tile)
    for t in svs.keys():
        out[t].append(np.mean(svs[t]))
        out[t].append(np.std(svs[t]))
        out[t].append(np.mean(tile[t]))
        out[t].append(np.std(tile[t]))

        out_v[t].append(np.mean(svs_v[t]))
        out_v[t].append(np.std(svs_v[t]))
        out_v[t].append(np.mean(tile_v[t]))
        out_v[t].append(np.std(tile_v[t]))
    book_=xlwt.Workbook()
    sheet=book_.add_sheet('summary bagging')
    sheet.write(0,0,'class')
    sheet.write(0,1,'train_WSI_mean_std')
    sheet.write(0,2,'train_patch_mean_std')
    sheet.write(0,3,'valid_WSI_mean_std')
    sheet.write(0,4,'valid_patch_mean_std')
    for i in range(len(out.keys())):
        c=list(out.keys())[i]
        sheet.write(i+1,0,c)
        sheet.write(i+1,1,str(out[c][0])+'_'+str(round(out[c][1],2)))
        sheet.write(i+1,2,str(out[c][2])+'_'+str(round(out[c][3])))
        sheet.write(i+1,3,str(out_v[c][0])+'_'+str(round(out_v[c][1])))
        sheet.write(i+1,4,str(out_v[c][2])+'_'+str(round(out_v[c][3])))
    book_.save(savepath)

def count_mean_std_xinjiang(xlspath,savepath):
    # cla=['BG','AAH','AIS','LPA','MIA']#
    # out=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # out_v=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # svs=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # tile=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # svs_v=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    # tile_v=dict(BG=[],AAH=[],AIS=[],LPA=[],MIA=[])
    cla=['BG','Acinar','LPA','Micropapillary','Papillary','Solid']#
    out=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    out_v=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    svs=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    tile=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    svs_v=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    tile_v=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    book=xlrd.open_workbook(xlspath)
    for i in range(10):
        sheet_train=book.sheet_by_name(str(i)+"_train")
        sheet_valid=book.sheet_by_name(str(i)+"_valid")
        for j in range(6):
            row=sheet_train.row_values(j+1)
            cla=row[0]
            n_svs=int(row[1])
            n_tile=int(row[2])
            svs[cla].append(n_svs)
            tile[cla].append(n_tile)

            row=sheet_valid.row_values(j+1)
            cla=row[0]
            n_svs=int(row[1])
            n_tile=int(row[2])
            svs_v[cla].append(n_svs)
            tile_v[cla].append(n_tile)
    print(tile,tile_v)
    for t in svs.keys():
        out[t].append(np.mean(svs[t]))
        out[t].append(np.std(svs[t],ddof=-1))
        out[t].append(np.mean(tile[t]))
        out[t].append(np.std(tile[t],ddof=-1))

        out_v[t].append(np.mean(svs_v[t]))
        out_v[t].append(np.std(svs_v[t],ddof=-1))
        out_v[t].append(np.mean(tile_v[t]))
        out_v[t].append(np.std(tile_v[t],ddof=-1))
    book_=xlwt.Workbook()
    sheet=book_.add_sheet('summary bagging')
    sheet.write(0,0,'class')
    sheet.write(0,1,'train_WSI_mean_std')
    sheet.write(0,2,'train_patch_mean_std')
    sheet.write(0,3,'valid_WSI_mean_std')
    sheet.write(0,4,'valid_patch_mean_std')
    for i in range(len(out.keys())):
        c=list(out.keys())[i]
        sheet.write(i+1,0,c)
        sheet.write(i+1,1,str(int(out[c][0]))+'_'+str(int(out[c][1])))
        sheet.write(i+1,2,str(int(out[c][2]))+'_'+str(int(out[c][3])))
        sheet.write(i+1,3,str(int(out_v[c][0]))+'_'+str(int(out_v[c][1])))
        sheet.write(i+1,4,str(int(out_v[c][2]))+'_'+str(int(out_v[c][3])))
    book_.save(savepath)

def count_wsi(wsi_path,json_path):
    wsi_n=len(os.listdir(wsi_path))
    json_n=len(os.listdir(json_path))
    print('wsi is {} json is {}'.format(wsi_n,json_n))
def count_maxiaomei_tile_in_23twodoctor(maxioamei_xls,tile_path,savepath):
    book=xlrd.open_workbook(maxioamei_xls)
    sheet=book.sheet_by_name('maxiaomei')
    wtbook=xlwt.Workbook()
    wksheet=wtbook.add_sheet('maxiaomei')
    wksheet.write(0,0,label='class')
    wksheet.write(0,1,label='patient_n')
    wksheet.write(0,2,label='tile')
    mxm_list=sheet.col_values(0)
    mag='5.0'
    blank=dict(LPA=[],Acinar=[],Papillary=[],Micropapillary=[],Solid=[],BG=[])
    i=0
    for c in os.listdir(tile_path):
        i += 1
        j=num=0
        wksheet.write(i,0,label=c)
        files_n=[]
        for f in os.listdir(os.path.join(tile_path,c)):
            fid=f.split('_files')[0]
            if os.path.splitext(f)[1]=='' and fid in mxm_list:
                if os.path.exists(os.path.join(tile_path,c,f,mag)) and os.listdir(os.path.join(tile_path,c,f,mag))!=[]:
                    #print(cl,fil,len(os.listdir(os.path.join(tile_root,cl,fil,'5.0'))))
                    num += len(os.listdir(os.path.join(tile_path,c,f,mag)))
                    j += 1
                    files_n.append(f+'_'+str(len(os.listdir(os.path.join(tile_path,c,f,mag)))))
                else:
                    #print(cl)
                    blank[c].append(f)
        wksheet.write(i,1,label=str(j))
        wksheet.write(i,2,label=str(num))
        wksheet.write(i,3,label=str(files_n))
    wtbook.save(savepath)
    print(blank)

if __name__ == "__main__":
    #cla=['LPA','Acinar','Papillary','Micropapillary','Solid']#,'BG'
    '''summary of exam in bagging'''
    # tile_root='/disk1/zhangyingxin/project/lung/daping_grade2/paper_script/'
    # out_='/disk1/zhangyingxin/project/lung/daping_grade2/paper_script/summary_train_valid_'

    # for i in range(10):
    #     mag='5.0'
    #     tile_root_=os.path.join(tile_root,str(i),'512px_tiled')
    #     out_path=out_+str(i)+'.xls'
    #     count_train_valid_tiles(tile_root_,out_path,mag)
    '''count mean and std of bagging tiles'''
    xlspath='/disk1/zhangyingxin/project/lung/daping_grade2/paper_script'
    savepath='/disk1/zhangyingxin/project/lung/daping_grade2/paper_script/table_pic_of_paper/train_valid_svs_tile.xls'

    #count_mean_std(xlspath,savepath)
    '''simple example summary'''
    tile_root_='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/256px_Tiled_exam_1_maxiaomei/'
    mag='5.0'
    out_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/256_exam_1_maxiaomei_tiles.xls'
    #count_train_valid_tiles(tile_root_,out_path,mag)
    #"/mnt/share/zhangyingxin/lung_local/xinjiang_2grade/paper_exam_try/work_space_all/liuting_mayuqing/20X_workspace/512px_tiled"
    #"/mnt/share/zhangyingxin/lung_local/xinjiang_2grade/paper_exam_try/work_space_all/liuting_mayuqing/512px_tiled/"
    '''xinjiang count all tiles'''
    tile_root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_preweight_deepslide/train'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/test_data_20fro5'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/512px_Tiled_maxiaomei_exam_limit_normaliaztion/train'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/512px_Tiled_maxiaomei_exam_limit_normaliztion/'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_overlap_20X'#_test
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled'
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/data/preweight_deepslide.xls'#_test
    #count_all_tiles(tile_root,savepath)
    '''为选择bagging 尽量均衡的train及valid的患者做参考'''
    tile_root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_overlap_20X'
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/data/all_tiles_patient_level_notest20X.xls'
    summaryf='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/data/patient_WSI_label_Ntiles20X.xls'
    #count_all_tiles_patient_level(tile_root,summaryf,savepath)
    '''summary of tiles in bagging xinjiang'''
    tile_root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment_v1/512px_Tiled_train_valid_bagging_patient_enhance'
    out_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment_v1/data/summary_train_valid_tiles_bagging.xls'
    mag='5.0'
    #count_train_valid_tiles_bagging(tile_root,out_path,mag,10)
    '''修改了进行WSI 级别统计'''
    tile_root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_all'
    out_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/data/all_tiles_wsi_level_notest.xls'
    #count_all_tiles_patient_level(tile_root,summaryf,savepath)
    '''新疆patients 级别bagging数据统计'''
    tile_root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_train_valid_bagging_patient_level'
    out_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/data/summary_train_valid_tiles_bagging_patient_level_add3rdata.xls'
    mag='5.0'
    #count_train_valid_tiles_bagging(tile_root,out_path,mag,10)
    '''count mean and std of xinjaing bagging tiles'''
    xlspath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/data/summary_train_valid_tiles_bagging_patient_level.xls'#_20Xfro5X
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/summary/train_valid_svs_tile_v1.xls'#_20X

    count_mean_std_xinjiang(xlspath,savepath)
    '''count doctor label的数量'''
    wsi_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/3rd_instalment/data/WSI'
    json_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/3rd_instalment/data/json'
    # json_path=['/mnt/data_backup/lung_cancer/XJF+XJZ/WSI_marked/第2批_public data_20210128/jiwenli_20210128/',
    # '/mnt/data_backup/lung_cancer/XJF+XJZ/WSI_marked/第2批_public data_20210128/maxiaomei_20210128/',]

    #count_wsi(wsi_path,json_path)
    '''get maxiaomei tile count'''
    maxioamei_xls='/disk1/zhangyingxin/project/lung/xinjiang_grade2/3rd_instalment/data/patient_WSI_label.xls'
    tile_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/512px_Tiled'
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/maxiaomei_tile_from_512px.xls'
    #count_maxiaomei_tile_in_23twodoctor(maxioamei_xls,tile_path,savepath)
    tile_root='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/512px_Tiled_maxiaomei'
    savepath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/all_tiles_patient_level_maxiaomei.xls'
    summaryf='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/data/patient_WSI_label_Ntiles20X.xls'
    #count_all_tiles_patient_level(tile_root,summaryf,savepath)
    '''xinjiaing maxioamei exam'''
    tile_root="/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/512px_Tiled_maxiaomei_exam_limit_normalization_bagging/"
    out_path="/disk1/zhangyingxin/project/lung/xinjiang_grade2/23rd_instalment_2doctor/data/summary_train_valid_tiles_bagging_maxiaomei.xls"
    mag='5.0'
    #count_train_valid_tiles_bagging(tile_root,out_path,mag,10)