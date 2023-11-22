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

visdrone_dir="$dataset_root_dir/VisDrone2019-MOT-$mode/"
seq_dirs=`ls $visdrone_dir/sequences`
for seq_dir in $seq_dirs
do
    if [ -d "$visdrone_dir/sequences/$seq_dir" ]
    then
        echo "Generating $seq_dir"
        python gt_visdrone_to_bdd100k.py --filepath "$visdrone_dir/annotations/$seq_dir.txt"
    fi 
done

