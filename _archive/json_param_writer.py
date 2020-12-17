import glob
import json
import numpy as np
import os
import easygui
import sys

#os.chdir('/media/bigdata/Abuzar_Data')
#file_list_path = '/media/bigdata/firing_space_plot/don_grant_figs/laser_files.txt'
# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    file_list = [dir_name]
else:
    file_list_path = '/media/fastdata/lfp_analyses/file_list.txt'
    file_list = open(file_list_path,'r').readlines()
    file_list = [x.rstrip() for x in file_list]

#file_list = glob.glob('**/**/**h5')
file_list.sort()
file_names = [os.path.basename(x) for x in file_list]
dir_names = [os.path.dirname(x) for x in file_list]
#fin_inds = [num for num,x in enumerate(file_names) if 'AM' in x]
selected_files = easygui.multchoicebox(msg = 'Please select files to run analysis on',
    choices = ['{}) '.format(num)+x for num,x in enumerate(file_names)])
fin_inds = [int(x.split(')')[0]) for x in selected_files]
fin_file_names = [file_names[x] for x in fin_inds]
fin_dir_names = [dir_names[x] for x in fin_inds]

type_16_16 = [list(map(int,np.arange(8)))+list(map(int,np.arange(24,32))),
                list(map(int, np.arange(8,24)))]
type_32_32 = [list(map(int, np.arange(32))),
                list(map(int, np.arange(32,64)))]
type_32 = [list(map(int,np.arange(32)))]
region_list = [["gc","bla"],["bla","gc"],["gc"],["bla"]]

def gen_json_dict(file_name):
    splits = file_name.split("_")
    global type_16_16, type_32_32
    electrode_type = easygui.buttonbox(msg = "Which electrode setup?"\
            "\n{}".format(file_name),
            choices = ("16-16", "32-32","32"))
    if electrode_type == "16-16":
        region_order = easygui.buttonbox(msg = "Which region order?"\
                "\n{}".format(file_name),
                choices = ("GC middle", "BLA middle"))
        electrode_list = type_16_16
        ans_ind = [ind_num for ind_num, region in \
                enumerate(["GC middle", "BLA middle"]) \
                if region == region_order][0]
        region_name_list = region_list[1-ans_ind] 
    elif electrode_type == "32-32":
        electrode_list = type_32_32
        region_order = easygui.buttonbox(msg = "Which region order?"\
                "\n{}".format(file_name),
                choices = ("GC first", "BLA first"))
        ans_ind = [ind_num for ind_num, region in \
                enumerate(["GC first", "BLA first"]) \
                if region == region_order][0]
        region_name_list = region_list[ans_ind] 
    elif electrode_type == "32":
        electrode_list = type_32
        region_order = easygui.buttonbox(msg = "Which region?"\
                "\n{}".format(file_name),
                choices = ("GC", "BLA"))
        region_name_list = region_list[int(2*(region_order == "GC")) +\
                                    int(3*(region_order == "BLA"))]

    region_dict = {'regions' : {region_name:electrodes \
            for region_name,electrodes in \
            zip(region_name_list, electrode_list)}}
    this_dict = {
            "name" : splits[0],
            "exp_type" : splits[1],
            "date": splits[2],
            "timestamp" : splits[3]}
    this_dict.update(region_dict)

    return this_dict

for dir_name, file_name in zip(fin_dir_names, fin_file_names):
    this_dict = gen_json_dict(file_name) 
    json_file_name = os.path.join(dir_name,
            '.'.join([file_name.split('.')[0],'json']))
    with open(json_file_name,'w') as file:
        json.dump(this_dict, file, indent = 4)
