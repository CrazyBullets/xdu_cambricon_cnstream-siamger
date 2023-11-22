# VisDrone2019-MOT
## Step 1. Prepare files list for VisDrone2019-MOT-val and VisDrone2019-MOT-test-dev
```
/workspace/dataset/public/zhumeng-dataset/VisDrone2019/VisDrone2019-MOT-val/sequences/uav0000086_00000_v.mp4
/workspace/dataset/public/zhumeng-dataset/VisDrone2019/VisDrone2019-MOT-val/sequences/uav0000117_02622_v.mp4
/workspace/dataset/public/zhumeng-dataset/VisDrone2019/VisDrone2019-MOT-val/sequences/uav0000137_00458_v.mp4
/workspace/dataset/public/zhumeng-dataset/VisDrone2019/VisDrone2019-MOT-val/sequences/uav0000182_00000_v.mp4
/workspace/dataset/public/zhumeng-dataset/VisDrone2019/VisDrone2019-MOT-val/sequences/uav0000268_05773_v.mp4
/workspace/dataset/public/zhumeng-dataset/VisDrone2019/VisDrone2019-MOT-val/sequences/uav0000305_00000_v.mp4
/workspace/dataset/public/zhumeng-dataset/VisDrone2019/VisDrone2019-MOT-val/sequences/uav0000339_00001_v.mp4
```
## Step 2. Prepare HOTA evaluation
```
git clone https://github.com/JonathonLuiten/TrackEval.git
```

Your directory tree should look like this:
```
visdrone_object_tracking
|-- README.md
|-- TrackEval
|-- config_template.json
|-- configs
|-- evaluate_HOTA.sh
|-- evaluate_MOT.py
|-- label_map_visdrone.txt
|-- output
|-- run.sh
`-- tools
```

## Step 3. Convert annotations format from VisDrone2019-MOT to MOT-Challenge
```
cd tools/to_motchallenge
./gen.sh val
./gen.sh test-dev
mv VisDrone2019-MOT* ../../TrackEval/data/gt/mot_challenge
mv seqmaps/* ../../TrackEval/data/gt/mot_challenge/seqmaps/
```

## Step 4. Run program
```
bash run.sh mlu270 test bytetrack test-dev
```

## Step 5. Run evaluation
```
python evaluate_MOT.py test-dev
```

or
```
bash evaluate_HOTA.sh bytetrack test-dev
```