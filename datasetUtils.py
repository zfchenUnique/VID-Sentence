import argparse
import cv2
import json
import copy
import os
import numpy as np
import shutil
import commands
TMP_DIR = '.tmp'
FFMPEG = 'ffmpeg'
SAVE_VIDEO = FFMPEG + ' -y -r %d -i %s/%s.jpg %s'


def jsonload(path):
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def imread_if_str(img):
    if isinstance(img, basestring):
        img = cv2.imread(img)
    return img

class BaseParser(argparse.ArgumentParser):
    def __init__(self, *arg_list, **arg_dict):
        super(BaseParser, self).__init__(*arg_list, **arg_dict)
        self.add_argument('--set_name', default='val', type=str)
        self.add_argument('--annFd', default='./data/ILSVRC', type=str)
        self.add_argument('--vis_gif_flag', action='store_true', default=False)

    def parse_args(self, *arg_list, **arg_dict):
        args = super(BaseParser, self).parse_args(*arg_list, **arg_dict)
        return args

def textread(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '').replace('\r', '')
    return lines

def resize_tube_bbx(tube_vis, frmImList_vis):
    for prpId, frm in enumerate(tube_vis):
        h, w, c = frmImList_vis[prpId].shape
        tube_vis[prpId][0] = tube_vis[prpId][0]*w
        tube_vis[prpId][2] = tube_vis[prpId][2]*w
        tube_vis[prpId][1] = tube_vis[prpId][1]*h
        tube_vis[prpId][3] = tube_vis[prpId][3]*h
    return tube_vis

def visTube_from_image(frmList, tube, outName):
    image_list   = list()
    for i, bbx in enumerate(tube):
        imName = frmList[i]
        img = draw_rectangle(imName, bbx)
        image_list.append(img)
        images2video(image_list, 10, outName)

def draw_rectangle(img, bbox, color=(0,0,255), thickness=3, use_dashed_line=False):
    img = imread_if_str(img)
    if isinstance(bbox, dict):
        bbox = [
            bbox['x1'],
            bbox['y1'],
            bbox['x2'],
            bbox['y2'],
        ]
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[0] = min(bbox[0], img.shape[1])
    bbox[1] = min(bbox[1], img.shape[0])
    bbox[2] = max(bbox[2], 0)
    bbox[3] = max(bbox[3], 0)
    bbox[2] = min(bbox[2], img.shape[1])
    bbox[3] = min(bbox[3], img.shape[0])
    assert bbox[2] >= bbox[0]
    assert bbox[3] >= bbox[1]
    assert bbox[0] >= 0
    assert bbox[1] >= 0
    assert bbox[2] <= img.shape[1]
    assert bbox[3] <= img.shape[0]
    cur_img = copy.deepcopy(img)
    #pdb.set_trace()
    if use_dashed_line:
        drawrect(
            cur_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            thickness,
            'dotted'
            )
    else:
        #pdb.set_trace()
        cv2.rectangle(
            cur_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            thickness)
    return cur_img


def images2video(image_list, frame_rate, video_path, max_edge=None):
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.mkdir(TMP_DIR)
    img_size = None
    for cur_num, cur_img in enumerate(image_list):
        cur_fname = os.path.join(TMP_DIR, '%08d.jpg' % cur_num)
        if max_edge is not None:
            cur_img = imread_if_str(cur_img)
        if isinstance(cur_img, str) or isinstance(cur_img, unicode):
            shutil.copyfile(cur_img, cur_fname)
        elif isinstance(cur_img, np.ndarray):
            max_len = max(cur_img.shape[:2])
            if max_len > max_edge and img_size is None and max_edge is not None:
                magnif = float(max_edge) / float(max_len)
                img_size = (int(cur_img.shape[1] * magnif), int(cur_img.shape[0] * magnif))
                cur_img = cv2.resize(cur_img, img_size)
            elif max_edge is not None:
                if img_size is None:
                    magnif = float(max_edge) / float(max_len)
                    img_size = (int(cur_img.shape[1] * magnif), int(cur_img.shape[0] * magnif))
                cur_img = cv2.resize(cur_img, img_size)
            cv2.imwrite(cur_fname, cur_img)
        else:
            NotImplementedError()
    print commands.getoutput(SAVE_VIDEO % (frame_rate, TMP_DIR, '%08d', video_path))
    shutil.rmtree(TMP_DIR)

def imread_if_str(img):
    if isinstance(img, basestring):
        img = cv2.imread(img)
    return img

