# Scripts for converting annotations format from VisDrone2019 to MOTChallenge
## Step 1. Specify root of dataset in gen.sh
```
dataset_root_dir=your_dataset_root
```

## Step 2. Choose mode of val or test-dev
```
bash gen.sh val
```

## Step 3. Move direcotries of annotations to TrackEval
```
mv seqmaps/* your_dir_of_TrackEval/data/gt/mot_challenge/seqmaps
mv VisDrone2019-MOT-val your_dir_of_TrackEval/data/gt/mot_challenge/
mv VisDrone2019-MOT-test-dev your_dir_of_TrackEval/data/gt/mot_challenge/
```