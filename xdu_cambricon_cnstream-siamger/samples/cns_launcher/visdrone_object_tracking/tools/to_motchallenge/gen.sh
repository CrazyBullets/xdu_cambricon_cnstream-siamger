dataset_root_dir="/workspace/dataset/public/zhumeng-dataset/VisDrone2019"

if [ ! -d $dataset_root_dir ]; then
    echo "Could not find VisDrone2019 in $dataset_root_dir"
    exit 1
fi

PrintUsages(){
    echo "Usages: gen.sh [val\test-dev]"
}

if [ $# -ne 1 ]; then
    PrintUsages
    exit 1
fi

if [[ ${1} != "val" && ${1} != "test-dev" ]]; then
    PrintUsages
    exit 1
fi

mode=${1}
mkdir -p seqmaps
# generate seqsmap
echo "name" > seqmaps/VisDrone2019-MOT-$mode.txt

visdrone_dir="$dataset_root_dir/VisDrone2019-MOT-$mode/"
seq_dirs=`ls $visdrone_dir/sequences`
for seq_dir in $seq_dirs
do
    if [ -d "$visdrone_dir/sequences/$seq_dir" ]
    then
        echo "Generating $seq_dir"
        mkdir -p "VisDrone2019-MOT-$mode/$seq_dir/gt"
        # generate seqinfo by ffprobe
        python catch_ffprobe.py --filepath "$visdrone_dir/sequences/$seq_dir.mp4"
        mv seqinfo.ini "VisDrone2019-MOT-$mode/$seq_dir"

        # relabel groundtrue format from VisDrone to MOT-Challenge
        python visdrone_to_mot.py --filepath "$visdrone_dir/annotations/$seq_dir.txt"
        mv gt.txt "VisDrone2019-MOT-$mode/$seq_dir/gt"

        # append seqsmap
        echo "$seq_dir" >> seqmaps/VisDrone2019-MOT-$mode.txt
    fi 
done

