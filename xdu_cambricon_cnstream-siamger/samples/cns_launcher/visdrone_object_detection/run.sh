#!/bin/bash
#************************************************************************************#
# @param
# src_frame_rate: frame rate for send data
# data_path: Video or image list path
# wait_time: When set to 0, it will automatically exit after the eos signal arrives
# loop = true: loop through video
#
# @notice: other flags see ${SAMPLES_ROOT}/bin/cns_launcher --help
#          when USB camera is the input source, please add 'usb' as the third parameter
#************************************************************************************#
CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE[0]});pwd)
CNSTREAM_ROOT=${CURRENT_DIR}/../../..
SAMPLES_ROOT=${CNSTREAM_ROOT}/samples
MODELS_ROOT=${CNSTREAM_ROOT}/data/models
CONFIGS_ROOT=${SAMPLES_ROOT}/cns_launcher/configs

PrintUsages(){
    echo "Usages: run.sh [mlu220/mlu270/mlu370] [encode_jpeg/encode_video/display/rtsp/test] $1"
}

if [[ $# == 3 ]];then
    if [[ ${3} == "usb" ]];then
        #change the input URL to usb device.
        echo "/dev/video0" > ${SAMPLES_ROOT}/files.list_video
    else
        PrintUsages "usb"
        exit 1
    fi
else
    if [[ $# != 2 ]];then
        PrintUsages
        exit 1
    fi
fi

if [[ ${2} != "encode_jpeg" && ${2} != "encode_video" && ${2} != "display" && ${2} != "rtsp"  && ${2} != "test" ]]; then
    PrintUsages
    exit 1
fi

# generate config file with selected sinker and selected platform
pushd ${CURRENT_DIR}
    if [[ $# == 2 ]];then
        sed 's/__PLATFORM_PLACEHOLDER__/'"${1}"'/g' config_template.json | sed 's/__SINKER_PLACEHOLDER__/'"${2}"'.json/g' &> config.json
    else
        #Because MLU may not support some usb cameras' codec format like AV_CODEC_ID_MSMPEG4V1, here we prefer to select CPU decoder.
        sed 's/decoder_type.*mlu/decoder_type\"\:\"cpu/g' ${CONFIGS_DIR}/decode_config.json &> ${CONFIGS_DIR}/cpu_decode_config.json
        sed 's/__PLATFORM_PLACEHOLDER__/'"${1}"'/g' config_template.json | sed 's/__SINKER_PLACEHOLDER__/'"${2}"'.json/g' | sed 's/decode_config/cpu_decode_config/g' &> config.json
    fi
popd

mkdir -p output
${SAMPLES_ROOT}/bin/cns_launcher  \
    --data_path ${SAMPLES_ROOT}/files.list_visdrone_image \
    --src_frame_rate -1 \
    --config_fname ${CURRENT_DIR}/config.json \
    --logtostderr=true
    
