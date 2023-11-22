CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE[0]});pwd)
CNSTREAM_ROOT=${CURRENT_DIR}/../../..
SAMPLES_ROOT=${CNSTREAM_ROOT}/samples
MODELS_ROOT=${CNSTREAM_ROOT}/data/models
CONFIGS_ROOT=${SAMPLES_ROOT}/cns_launcher/configs

# Supported platform
Platforms=(
    mlu220
    mlu270
)
# Supported sinkers
Sinkers=(
    encode_jpeg
    encode_video
    display
    rtsp
    test
)
# Supported trackers
Trackers=(
    sort
    deepsort
    bytetrack
    bottrack
    strongsort
    ocsort
)
# Supported subsets
Subsets=(
    val
    test-dev
    debug
)

PrintUsages(){
    echo "Usages: run.sh [${Platforms[@]}] [${Sinkers[@]}] [${Trackers[@]}] [${Subsets[@]}]"
}

if [ $# -ne 4 ]; then
    PrintUsages
    exit 1
fi

if [[ ! "${Platforms[@]}" =~ ${1} ]]; then
    echo "Unsupported platforms: ${1}"
    PrintUsages
    exit 1
fi

if [[ ! "${Sinkers[@]}" =~ ${2} ]]; then
    echo "Unsupported platforms: ${2}"
    PrintUsages
    exit 1
fi

if [[ ! "${Trackers[@]}" =~ ${3} ]]; then
    echo "Unsupported platforms: ${3}"
    PrintUsages
    exit 1
fi

if [[ ! "${Subsets[@]}" =~ ${4} ]]; then
    echo "Unsupported platforms: ${4}"
    PrintUsages
    exit 1
fi


# generate config file with selected sinker and selected platform
pushd ${CURRENT_DIR}
    sed 's/__PLATFORM_PLACEHOLDER__/'"${1}"'/g;s/__NN__/'"${3}"'/g' config_template.json | sed 's/__SINKER_PLACEHOLDER__/'"${2}"'.json/g' &> config.json
popd

mkdir -p output
rm -f output/*

if [[ ${4} == "debug" ]]; then
export EDK_LOG_FILTER=OCSORT:4
gdb --args \
${SAMPLES_ROOT}/bin/cns_launcher  \
    --data_path ${SAMPLES_ROOT}/files.list_visdrone_${4} \
    --src_frame_rate 1  \
    --config_fname ${CURRENT_DIR}/config.json \
    --log_to_stderr=true \
    --perf_level=2 \
    --trace_data_dir=output
else
export EDK_LOG_FILTER=
${SAMPLES_ROOT}/bin/cns_launcher  \
    --data_path ${SAMPLES_ROOT}/files.list_visdrone_${4} \
    --src_frame_rate -1  \
    --config_fname ${CURRENT_DIR}/config.json \
    --log_to_stderr=true \
    --perf_level=2 \
    --trace_data_dir=output
fi


