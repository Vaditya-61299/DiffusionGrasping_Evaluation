import os
file_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
new_split_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),"new_splits")


object_classes=["Mug","CerealBox","WineBottle","Plant","Knife","PSP","Spoon","Camera","WineGlass","Teacup","ToyFigure","Hat","Book","Ring","Hammer"]        #Put the name of class in this list that you want to test

from DiffusionFields.se3dif.datasets import AcronymGraspsDirectory
from DiffusionFields.se3dif.utils import get_data_src
from DiffusionFields.se3dif.utils import to_numpy, to_torch, get_grasps_src
import glob
import json
from pathlib import Path


for obj_class in object_classes:
    list=[]
    filename=os.path.join(file_path,"grasps")
    grasps_files = sorted(glob.glob(filename + '/' + obj_class + '/*.h5'))
    for grasp_file in grasps_files:
        list.append(grasp_file)
    list=[line.strip().replace(".h5",".json") for line in list]
    list=[Path(line.strip()).name for line in list]
    with open (os.path.join(new_split_path,f"{obj_class}.json" , "a") as file:
        _result={}
        _result["test"]=list
        json.dump(_result,file,indent=4,default=str)




