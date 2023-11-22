import json
import os

from glob import glob
from pathlib import Path

RESULT_PREFIX = './output'

id2label = {'0':"pedestrian",
            '1':"people",
            '2':"bicycle",
            '3':"car",
            '4':"van",
            '5':"truck",
            '6':"tricycle",
            '7':"awning-tricycle",
            '8':"bus",
            '9':"motor"}

if __name__ == "__main__":
    
    visdrone_files = glob(os.path.join(RESULT_PREFIX, "*.txt"))

    for visdrone_file in visdrone_files:
        video_name = Path(visdrone_file).stem
        print("Converting {}".format(video_name))
        bdd100k_rst = list()

        with open(visdrone_file, 'r') as vis_f:
            lines = vis_f.readlines()
            pre_frame_index = 1 # start from frame 1
            labels = list()
            for line in lines:
                cur_frame_index, target_id, \
                bb_left, bb_top, bb_width, bb_height, \
                score, obj_category, truncation, occlusion = line.split(",") 

                cur_frame_index = int(cur_frame_index)

                if pre_frame_index != cur_frame_index:
                    rst = {"video_name" : video_name,
                        "name" : "{}/{:0>7d}.jpg".format(video_name, pre_frame_index),
                        "index" : pre_frame_index - 1,
                        "labels" : labels,
                    }
                    bdd100k_rst.append(rst)
                    labels = list()

                labels.append({"category" : id2label[obj_category],
                                "id" : int(target_id),
                                "box2d":{
                                    "x1": float(bb_left),
                                    "y1": float(bb_top),
                                    "x2": float(bb_left) + float(bb_width),
                                    "y2": float(bb_top) + float(bb_height)
                                },
                                "score" : float(score)})
                
                pre_frame_index = cur_frame_index
                
            rst = {"name" : "{:0>7d}.jpg".format(pre_frame_index),
                    "video_name" : video_name,
                    "index" : pre_frame_index - 1,
                    "labels" : labels,
                }
            bdd100k_rst.append(rst)

        bdd100k_file = visdrone_file.replace("txt","json")
        with open(bdd100k_file, "w") as bdd_f:
            json.dump(bdd100k_rst, bdd_f, indent=4)