import sys
sys.path.append('../')
sys.path.append('../util')
import pdb
import h5py
import csv
import numpy as np

import matplotlib
import random
import math
import os
from datasetUtils import *
import copy

class vidInfoParser(object):
    def __init__(self, set_name, annFd):
        self.tube_gt_path = os.path.join(annFd, 'Annotations/VID/tubeGt', set_name)
        self.tube_name_list_fn = os.path.join(annFd, 'Data/VID/annSamples/', set_name+'_valid_list.txt')
        self.jpg_folder = os.path.join(annFd, 'Data/VID/', set_name)
        self.info_lines = textread(self.tube_name_list_fn)
        self.set_name = set_name
        self.tube_ann_list_fn = os.path.join(annFd, 'Data/VID/annSamples/', set_name + '_ann_list_v2.txt')
        #pdb.set_trace()
        ins_lines = textread(self.tube_ann_list_fn)
        ann_dict_set_dict = {}
        for line in ins_lines:
            ins_id_str, caption = line.split(',', 1)
            ins_id = int(ins_id_str)
            if ins_id not in ann_dict_set_dict.keys():
                ann_dict_set_dict[ins_id] = list()
            ann_dict_set_dict[ins_id].append(caption)
        self.tube_cap_dict = ann_dict_set_dict

    def get_length(self):
        return len(self.info_lines)

    def get_shot_info_from_index(self, index):
        info_Str = self.info_lines[index]
        vd_name, ins_id_str = info_Str.split(',')
        return vd_name,  ins_id_str

    def get_shot_anno_from_index(self, index):
        vd_name, ins_id_str =  self.get_shot_info_from_index(index)
        jsFn = os.path.join(self.tube_gt_path, vd_name + '.js')
        annDict = jsonload(jsFn)
        ann = None
        for ii, ann in enumerate(annDict['annotations']):
            track = ann['track']
            trackId = ann['id']
            if(trackId!=ins_id_str):
                continue
            break;
        return ann, vd_name

    def get_shot_frame_list_from_index(self, index):
        ann, vd_name = self.get_shot_anno_from_index(index)
        frm_list = list()
        track = ann['track']
        trackId = ann['id']
        frmNum = len(track)
        for iii in range(frmNum):
            vdFrmInfo = track[iii]
            imPath = '%06d' %(vdFrmInfo['frame']-1)
            frm_list.append(imPath)
        return frm_list, vd_name

    def proposal_path_set_up(self, prpPath):
        self.propsal_path = os.path.join(prpPath, self.set_name)

def demo_for_dataset(opts):
    '''
    A demo for using the dataset annotation
    '''
    vid_parser = vidInfoParser(opts.set_name, opts.annFd) 
    # get length for the dataset
    set_length = vid_parser.get_length()
    print('The VID %s set contains %d instances\n' %(opts.set_name, set_length))
    use_key_index = vid_parser.tube_cap_dict.keys()
    use_key_index.sort()
    # visualize the first example
    vis_id = 0
    index = use_key_index[vis_id]
    caption_str = vid_parser.tube_cap_dict[index][0]
    ann, vd_name = vid_parser.get_shot_anno_from_index(index)
    frm_list, vd_name = vid_parser.get_shot_frame_list_from_index(index)

    print('Visualizing the %d-th example\n' %(vis_id))
    print(caption_str) 
    
    if opts.vis_gif_flag:
        print('Generating GIF for the instance.\n')
        frmImNameList = [os.path.join(vid_parser.jpg_folder, vd_name, frame_name + '.JPEG') for frame_name in frm_list]
        #pdb.set_trace()        
        frmImList = list()
        tube = list()
        for fId, imPath  in enumerate(frmImNameList):
            img = cv2.imread(imPath)
            frmImList.append(img)
            bbox = ann['track'][fId]['bbox']
            tube.append(bbox)
        vis_frame_num = 30
        visIner =max(int(len(frmImList) /vis_frame_num), 1)
        frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
        tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
        visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis, vd_name+'.gif')
        print('Finishing generating GIF for the instance.\n')
        print(caption_str) 

if __name__ == '__main__':
    parser = BaseParser()
    opts = parser.parse_args()
    demo_for_dataset(opts) 

