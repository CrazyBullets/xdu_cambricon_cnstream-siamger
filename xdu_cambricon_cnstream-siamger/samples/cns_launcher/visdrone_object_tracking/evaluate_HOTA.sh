TrackEval_dir="./TrackEval"

# Supported trackers
Trackers=(
    sort
    deepsort
    bytetrack
    strongsort
    ocsort
)

PrintUsages(){
    echo "Usages: evaluate_HOTA.sh [${Trackers[@]}] [val/test-dev]"
}

if [ $# -ne 2 ]; then
    PrintUsages
    exit 1
fi

if [[ ! "${Trackers[@]}" =~ ${1} ]]; then
    echo "Unsupported platforms: ${1}"
    PrintUsages
    exit 1
fi

if [[ ${2} != "val" && ${2} != "test-dev" && ${2} ]]; then
    PrintUsages
    exit 1
fi

data_dir="$TrackEval_dir/data/trackers/mot_challenge/VisDrone2019-MOT-${2}/${1}/data"
mkdir -p $data_dir
mv output/uav*.txt $data_dir

cd $TrackEval_dir
python scripts/run_mot_challenge.py \
--BENCHMARK VisDrone2019-MOT \
--SPLIT_TO_EVAL ${2} \
--TRACKERS_TO_EVAL ${1} \
--METRICS HOTA CLEAR Identity \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 20 \
--DO_PREPROC False

cd -