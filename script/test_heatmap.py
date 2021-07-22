import os
#import xlrd
import shutil
import json
from optparse import OptionParser

def heatmap(test_path,stitch_path,heatmap_path):
	tp=test_path
	st=stitch_path
	hp=heatmap_path
	if not os.path.exists(hp):
		os.mkdir(hp)
	os.system("python "+DeepPATH+"/03_postprocessing/heatmap_wholeimg_alpha.py --TileData "+tp+'/out_filename_Stats.txt --StitchedFolder '+st+' --OutputFolder '+hp)
def heatmap_rightlabel(jsonpath,test_path,stitch_path,heatmap_path,extr):
	jp=jsonpath
	tp=test_path
	st=stitch_path
	hp=heatmap_path

	if not os.path.exists(hp):
		os.mkdir(hp)
	print(extr)
	os.system("python "+DeepPATH+"/03_postprocessing/heatmap_whole_alpha_polygon_cv2.py --rightlabel "+jp+" --TileData "+tp+'/out_filename_Stats.txt --StitchedFolder '+st+' --OutputFolder '+hp+' --Extr '+extr)

def main():
	
	'''generate heatmap
	heat_path = os.path.join(work_path, 'heatmap5.0_rightlabel_cv2_lcolor')
	stitch_path = os.path.join(work_path, 'WSI_stitch')
	test_result_path = os.path.join(work_path, 'result','test', 'test_960k')
	#heatmap(test_result_path, stitch_path, heat_path)
	heatmap_rightlabel(jsonpath,test_result_path, stitch_path, heat_path)'''
	'''generate heatmap bagging xinjiang'''
	# heat_path = os.path.join(work_path,'train_valid_test' ,'heatmap','heatmap5.0_rightlabel_cv2_TileNoLabel_paper_3class_last_nonormal')
	# stitch_path = os.path.join(work_path,'data', 'WSI_stitch','train')
	# test_result_path ='/disk1/zhangyingxin/project/lung/xinjiang_grade2/min_rely/WSI_space_paper/result/3_class_paper_solid/test_230000k' 
	heat_path = os.path.join(work_path,'train_valid_test' ,'heatmap','heatmap5.0_rightlabel_cv2_TileNoLabel')
	stitch_path = os.path.join(work_path,'data', 'WSI_stitch','a')#os.path.join(work_path,'data', 'WSI_stitch','train')
	test_result_path ='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_patient_level/9/result/test_tile_nolabel/test_2268k' 

	#os.path.join(work_path,'train_valid_test','bagging_patient_level','9', 'result','test_trained', 'test_2268k')
	#heatmap(test_result_path, stitch_path, heat_path)
	#print(jsonpath,test_result_path, stitch_path, heat_path)
	extr='svs'
	#heatmap(test_result_path,stitch_path,heat_path)
	heatmap_rightlabel(jsonpath,test_result_path, stitch_path, heat_path,extr)
if __name__ == "__main__":
	'''xinjaing '''
	# work_path='/mnt/share/zhangyingxin/lung_local/xinjiang_2grade/paper_exam_try/work_space_all/liuting_mayuqing'
	# DeepPATH="/mnt/share/zhangyingxin/lung_local/DeepPATH_code"
	# jsonpath='/mnt/share/zhangyingxin/lung_local/xinjiang_2grade/paper_exam_try/data/json'
	'''daping'''
	# work_path='/disk1/zhangyingxin/project/lung/daping_grade2/Exam_1'
	# DeepPATH="/mnt/share/zhangyingxin/lung_local/DeepPATH_code"
	# jsonpath='/mnt/data_backup/lung_cancer/DP/WSI_marked/Daping_20201218/'
	'''xinjaing paper bagging'''
	work_path='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment'
	DeepPATH="/mnt/share/zhangyingxin/lung_local/DeepPATH_code"
	jsonpath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/data/json'

	main()

