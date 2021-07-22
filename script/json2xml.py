import json
from xml.dom.minidom import Document
import xmltodict
import os
def jsonToXml(json_str):
    try:
        xml_str=""
        xml_str = xmltodict.unparse(json_str, encoding='utf-8')
    except:
        xml_str = xmltodict.unparse({'request': json_str}, encoding='utf-8')
    finally:
        return xml_str
def json_to_xml(json_path,xml_path):
    if(os.path.exists(xml_path)==False):
        os.makedirs(xml_path)
    di = os.listdir(json_path)
    for file in di:
        file_list=file.split(".")
        with open(os.path.join(json_path,file), 'r') as load_f:
            load_dict = json.load(load_f)
        json_result = jsonToXml(load_dict)
        f = open(os.path.join(xml_path,file_list[0]+".xml"), 'w', encoding="UTF-8")
        f.write(json_result)
        f.close()
def xmlTojson(json_path,xml_path):
    # convertedDict = xmltodict.parse(xmlStr)
    # jsonStr = json.dumps(convertedDict)
    if(os.path.exists(json_path)==False):
        os.makedirs(json_path)
    di = os.listdir(xml_path)
    for file in di:
        file_list=file.split(".")
        xml_file = open(xml_path, 'r')
        xml_str = xml_file.read()
        # 将读取的xml字符串转换为字典
        json_dict = xmltodict.parse(xml_str)
        # 将字典转换为json格式的字符串
        json_str = json.dumps(json_dict, indent = 2)
        # with open(os.path.join(json_path,file), 'r') as load_f:
        #     load_dict = json.load(load_f)
        # json_result = jsonToXml(load_dict)
        # with open('res.json', 'w',encoding='utf-8') as f:
        #     f.write(json_1)
        f = open(os.path.join(json_path,file_list[0]+".json"), 'w', encoding="UTF-8")
        f.write(json_str)
        f.close()


# xml to json
def xmlToJson(xml_str):
    try:
        json_dict = xmltodict.parse(xml_str, encoding = 'utf-8')
        json_str = json.dumps(json_dict, indent = 2)
        return json_str
    except:
        pass
  
# json to xml
def jsonToXml1(json_str):
    try:
        json_dict = json.loads(json_str)
        xml_str = xmltodict.unparse(json_dict, encoding = 'utf-8')
    except:
        xml_str = xmltodict.umparse({'request':json_dict}, encoding = 'utf-8')
    finally:
        return xml_str
  
# load xml file
def load_json(xml_path):
    # 获取xml文件
    xml_file = open(xml_path, 'r')
    xml_str = xml_file.read()
    # 将读取的xml字符串转换为字典
    json_dict = xmltodict.parse(xml_str)
    # 将字典转换为json格式的字符串
    json_str = json.dumps(json_dict, indent = 2)
    return json_str
# json_1=load_json( 'D:\crazing_1.xml')
# with open('res.json', 'w',encoding='utf-8') as f:
#     f.write(json_1)
# def traverse_data(json_data,preNode = None):
#     # creat root 
#     if isinstance(json_data,dict):
#         dict_data= {}
#         #get  all key value
#         keys = json_data.keys()
#         node = doc.createElement("node")
#         for var in keys:
#             if var != "children":
#                 node.setAttribute(str(var), str(json_data[var]))
#         if preNode != None:
#             preNode.appendChild(node)
#         else:
#             doc.appendChild(node)
#         #creat root element 
#         if "children" in json_data and json_data.get("children") != None:
#             traverse_data(json_data.get("children"),node)

#     elif isinstance(json_data,list):
#         for element in json_data:
#             if element != None and len(element) > 0:
#                 if isinstance(element,dict):
#                     dict_data1 = {}
#                     #get all key value 
#                     keys1 = element.keys()
#                     node = doc.createElement("node")
#                     for var1 in keys1:
#                         if var1 != "children":
#                             print(element[var1])
#                             node.setAttribute(str(var1),str(element[var1]))
#                     if preNode != None:
#                         preNode.appendChild(node)
#                     else:
#                         doc.appendChild(node)
#                     if element.has_key("children") and element.get("children") != None:
#                         traverse_data(element.get("children"),node)
if __name__ =='__main__':
    # src = open(u"/mnt/share/zhangyingxin/lung_local/daping_2grade/pre_exam/json/1937646,2-20200312172223.TMAP.json")
    # obj = json.loads(src.read())
    # doc = Document()
    # traverse_data(obj)
    # fp = open("/mnt/share/zhangyingxin/lung_local/daping_2grade/pre_exam/xml/1937646,2-20200312172223.TMAP.xml", 'w')
    # doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    #"/home/gengxiaoqi/Hygieia_singularity/applications/Ki67_ER_PR/ki67_test_result/NEC_M004_03_zyx/json2xml/"
    #"/mnt/share/zhangyingxin/lung_local/daping_2grade/pre_exam/data/json/maqiang_2020_10_13/"
    #'/mnt/share/zhangyingxin/lung_local/daping_2grade/pre_exam/data/json_close/'
    #"/home/gengxiaoqi/Hygieia_singularity/applications/Ki67_ER_PR/ki67_test_result/NEC_M004_03_zyx/json2xml/"
    #'/mnt/share/zhangyingxin/lung_local/daping_2grade/pre_exam/data/xml/'
    '''xinjaing data structure
    json_path="/mnt/share/zhangyingxin/lung_local/xinjiang_2grade/paper_exam_try/data/json/mayuqing_group2/"
    xml_path="/mnt/share/zhangyingxin/lung_local/xinjiang_2grade/paper_exam_try/data/xml/mayuqing_group2/"
    json_to_xml(json_path,xml_path)
    '''
    '''daping data structure
    json_='/mnt/data_backup/lung_cancer/DP/WSI_marked/Daping_20201218/'
    xml='/mnt/share/zhangyingxin/lung_local/daping_2grade/Exam_1/data/xml'
    for j in os.listdir(json_):
        json_to_xml(os.path.join(json_,j),xml)
    '''
    '''UCSF'''
    # json_='/home/yxzhang/ki67/Hygieia/applications/Ki67_ER_PR/test_result/patch_test/json'
    # xml='/home/yxzhang/ki67/Hygieia/applications/Ki67_ER_PR/test_result/patch_test/xml/to_json/'
    # xmlTojson(json_,xml)
    '''lung grade2'''
    json_='/disk1/zhangyingxin/project/lung/xinjiang_grade2/3rd_instalment/data/json'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment_v1/data/json'
    xml='/disk1/zhangyingxin/project/lung/xinjiang_grade2/3rd_instalment/data/xml'
    json_to_xml(json_,xml)
