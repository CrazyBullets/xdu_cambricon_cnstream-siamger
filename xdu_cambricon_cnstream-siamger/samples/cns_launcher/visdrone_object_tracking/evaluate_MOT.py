"""
evaluate tracking results 
"""

import enum
import os
import numpy as np
import copy
# from compute_mIOU import main
import motmetrics as mm
import argparse
from glob import glob
from pathlib import Path
mm.lap.default_solver = 'lap'


GT_PREFIX = os.path.join('/workspace/dataset/public/zhumeng-dataset/VisDrone2019/VisDrone2019-MOT-val/annotations')
RESULT_PREFIX = './output'

class SeqEvaluator:
    def __init__(self, seq_name, gt_name, data_type='visdrone', ignore_cls_idx=set()) -> None:
        """
        create a evaluator for each class

        seq_name: name of the sequence
        gt_name: name of the gt sequence
        data_type: data format, currently support 'visdrone' and 'mot'
        ignore_cls_idx: set, the class of object ignored
        """
        self.seq_name = seq_name
        self.data_type = data_type

        self.ignore_cls_idx = ignore_cls_idx

        if self.data_type == 'visdrone':
            self.valid_cls_idx = {i for i in range(1, 11)} - self.ignore_cls_idx
        elif self.data_type == 'mot':
            self.valid_cls_idx = {i for i in range(1, 12)} - self.ignore_cls_idx
        else:
            raise NotImplementedError


        self.gt_frame_dict = self.read_result(gt_name, is_gt=True)
        self.gt_ignore_frame_dict = self.read_result(gt_name, is_ignore=True)       

        self.acc = mm.MOTAccumulator(auto_id=True)  # 初始化评估类

    def read_result(self, seq_name, is_gt=False, is_ignore=False) -> dict:
        """
        将结果转换为字典
        """

        result_dict = dict()
        if is_gt or is_ignore:
            seq_name = os.path.join(GT_PREFIX, seq_name)
        else:
            seq_name = os.path.join(RESULT_PREFIX, seq_name)
        # seq_name = os.path.join(RESULT_PREFIX, seq_name) if not is_gt else os.path.join(GT_PREFIX, seq_name)
        print(seq_name)
        with open(seq_name, 'r') as f:
            for line in f.readlines():
                line = line.replace(' ', ',')

                linelist = line.split(',')
                fid = int(linelist[0])  # 帧id
                result_dict.setdefault(fid, list())

                if is_gt:
                    label = int(float(linelist[7]))
                    mark = int(float(linelist[6]))
                    if mark == 0 or label not in self.valid_cls_idx:
                        continue
                    
                    score = 1
                elif is_ignore:
                    label = int(float(linelist[7]))
                    if self.data_type == 'mot':
                        vis_ratio = float(linelist[8]) 
                    elif self.data_type == 'visdrone':
                        vis_ratio = 1 - float(linelist[8]) / 3
                    else:
                        raise NotImplementedError
                    if label not in self.ignore_cls_idx and vis_ratio >= 0:
                        continue
                    score = 1
                else:
                    score = -1

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(float(linelist[1]))


                result_dict[fid].append((tlwh, target_id, score))
                
            f.close()
        return result_dict

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids) -> None:
        """
        The core function evaluates the metrics of a frame

        Frame_id: int, the frame ordinal number of the current frame
        trk_tlwhs: tuple, coordinate top-left width-height
        trk_ids: int, target ID
        """
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = self.unzip_objs(gt_objs)[:2]  # gt_tlwhs: np.ndarray(dtype=float)

        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = self.unzip_objs(ignore_objs)[0]


        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # IoU matching
        # TODO: more concise method
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        self.acc.update(gt_ids, trk_ids, iou_distance)

    def eval_seq(self) -> mm.MOTAccumulator:
        self.acc = mm.MOTAccumulator(auto_id=True)
        result_frame_dict = self.read_result(self.seq_name, is_gt=False)

        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))  # 取结果和真值的帧的并集

        for frame_id in frames:  # 对每一帧进行评估
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = self.unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids)

        return self.acc
    
    def unzip_objs(self, objs):
        if len(objs) > 0:
            tlwhs, ids, scores = zip(*objs)
        else:
            tlwhs, ids, scores = [], [], []
        tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

        return tlwhs, ids, scores


def evaluate(result_files, gt_files, data_type, result_folder=''):
    """
    result_files: List[str], format: frame_id, track_id, x, y, w, h, conf
    gt_files:  List[str],  
    data_type: str, data format,  visdrone mot
    result_folder: if result files is under a folder, then add to result prefix
    """
    assert len(result_files) == len(gt_files)

    accs = []

    for idx, result_f in enumerate(result_files):
        gt_f = gt_files[idx]  # 对应的真值文件

        evaluator = SeqEvaluator(seq_name=os.path.join(result_folder, result_f), gt_name=gt_f, data_type=data_type)
    # evaluator = SeqEvaluator(seq_name=result_files, gt_name=gt_files, data_type=data_type)
        accs.append(evaluator.eval_seq())
    
    # 得到总指标
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=result_files,
            generate_overall=True
        )
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

def get_args():
    """
    This function creates a command-line argument parser and returns parsed arguments

    :returns: parsed command-line arguments (in the form of argparse.Namespace)
    """
    parser = argparse.ArgumentParser(description='Calculate the mAP of coco dataset')

    parser.add_argument("--result_files", dest = 'result_files', help =
                        "path of result files",
                        default = '', type = str)
    parser.add_argument("--gt_files", dest = "gt_files", help =
                        "path of groundtruth files",
                        default = '',type = str)
    parser.add_argument("--data_type", dest = "data_type", help =
                        "The data type. visdrone | mot",
                        default = 'mot',type = str)
    parser.add_argument("--result_folder", dest = "result_folder", help =
                        "name of the output file",
                        default = '',type = str)
    parser.add_argument("--mode","-m", dest = "mode", choices=["val","test-dev"], help =
                        "val or test-dev mode",
                        default = 'val',type = str)                
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    result_files = args.result_files
    gt_files = args.gt_files
    data_type = args.data_type
    result_folder = args.result_folder

    # result_files = ["uav0000086_00000_v.txt","uav0000117_02622_v.txt","uav0000137_00458_v.txt","uav0000182_00000_v.txt","uav0000268_05773_v.txt","uav0000305_00000_v.txt","uav0000339_00001_v.txt"]
    # gt_files = ["uav0000086_00000_v.txt","uav0000117_02622_v.txt","uav0000137_00458_v.txt","uav0000182_00000_v.txt","uav0000268_05773_v.txt","uav0000305_00000_v.txt","uav0000339_00001_v.txt"]
    GT_PREFIX = GT_PREFIX.replace("val", args.mode)
    result_files = glob(os.path.join(RESULT_PREFIX, "*.txt"))
    result_files = [ Path(f).name for f in result_files]

    evaluate(result_files, result_files, data_type, result_folder)

# python evaluate.py --result_files uav0000086_00000_v.txt --gt_files uav0000086_00000_v.txt