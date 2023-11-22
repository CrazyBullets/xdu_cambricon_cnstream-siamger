import json
import os
import argparse

from glob import glob
from pathlib import Path

RESULT_PREFIX = './output'

id2label = {'0':"ignored",
            '1':"pedestrian",
            '2':"people",
            '3':"bicycle",
            '4':"car",
            '5':"van",
            '6':"truck",
            '7':"tricycle",
            '8':"awning-tricycle",
            '9':"bus",
            '10':"motor",
            '11':"others"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert annotations format from VisDrone2019 to BDD100k")
    parser.add_argument("--filepath", dest = 'filepath', help =
                        "path of annotations files",
                        required=True, type = str)

    args = parser.parse_args()

    visdrone_file = args.filepath

    video_name = Path(visdrone_file).stem

    bdd100k_rst = list()

    with open(visdrone_file, 'r') as vis_f:
        lines = vis_f.readlines()
        lines.sort(key = lambda line : (int(line.split(",")[0])))
        pre_frame_index = 1 # start from frame 1
        labels = list()
        for line in lines:
            cur_frame_index, target_id, \
            bb_left, bb_top, bb_width, bb_height, \
            score, obj_category, truncation, occlusion = line.split(",") 

            cur_frame_index = int(cur_frame_index)

            if pre_frame_index != cur_frame_index:
                rst = {"name" : "{:0>7d}.jpg".format(pre_frame_index),
                    "video_name" : video_name,
                    "index" : pre_frame_index - 1,
                    "labels" : labels,
                }
                bdd100k_rst.append(rst)
                labels = list()

            labels.append({"category" : id2label[obj_category],
                            "id" : int(target_id),
                            "attributes" :{
                                "Occluded":  occlusion == 1,
                                "Truncated" : truncation != 0,
                                "Crowd": False
                            },
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

    bdd100k_file = os.path.join("./","{}.json".format(video_name))
    with open(bdd100k_file, "w") as bdd_f:
        json.dump(bdd100k_rst, bdd_f, indent=4)