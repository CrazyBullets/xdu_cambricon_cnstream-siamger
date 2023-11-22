#!/bin/bash
#************************************************************************************#
# @param
# src_frame_rate: frame rate for send data
# data_path: Video or image list path
# wait_time: When set to 0, it will automatically exit after the eos signal arrives
# loop = true: loop through video
#
# @notice: other flags see ${SAMPLES_DIR}/bin/cns_launcher --help
#          when USB camera is the input source, please add 'usb' as the third parameter
#************************************************************************************#
CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE[0]});pwd)
CNSTREAM_ROOT=${CURRENT_DIR}/../../..
SAMPLES_ROOT=${CNSTREAM_ROOT}/samples
MODELS_ROOT=${CNSTREAM_ROOT}/data/models
CONFIGS_ROOT=${SAMPLES_ROOT}/cns_launcher/configs

PrintUsages(){
    echo "Usages: run.sh [mlu220/mlu270] [encode_jpeg/encode_video/display/rtsp/kafka] [rtsp/udp]$1"
    echo "[mlu220/mlu270] : MLU device"
    echo "[encode_jpeg/encode_video/display/rtsp/kafka] : Output format"
    echo "[rtsp/udp/udp2/udp3] : Input format"
}

${SAMPLES_ROOT}/generate_file_list.sh

if [[ $# != 3 ]];then
    PrintUsages
    exit 1
fi

if [[ ${1} == "mlu220" ]]; then
    MODEL_PATH=${MODELS_ROOT}/bdd100k_yolov5l_1_1_MLU220.cambricon
    # REMOTE_MODEL_PATH=http://video.cambricon.com/models/MLU220/yolov3_b4c4_argb_mlu220.cambricon
elif [[ ${1} == "mlu270" ]]; then
    MODEL_PATH=${MODELS_ROOT}/yolov3_b4c4_argb_mlu270.cambricon
    # REMOTE_MODEL_PATH=http://video.cambricon.com/models/MLU270/yolov3_b4c4_argb_mlu270.cambricon
else
    PrintUsages
    exit 1
fi

if [[ ${2} != "encode_jpeg" && ${2} != "encode_video" && ${2} != "display" && ${2} != "rtsp"  && ${2} != "kafka" ]]; then
    PrintUsages
    exit 1
fi

if [[ ${3} != "rtsp" && ${3} != "udp" && ${3} != "udp2" && ${3} != "udp3" ]]; then
    PrintUsages
    exit 1
fi

LABEL_PATH=${MODELS_ROOT}/label_map_bdd100k.txt
# REMOTE_LABEL_PATH=http://video.cambricon.com/models/labels/label_map_coco.txt

mkdir -p ${MODELS_ROOT}

if [[ ! -f ${MODEL_PATH} ]]; then
    # wget -O ${MODEL_PATH} ${REMOTE_MODEL_PATH}
    # if [ $? -ne 0 ]; then
    #     echo "Download ${REMOTE_MODEL_PATH} to ${MODEL_PATH} failed."
    #     exit 1
    # fi
    echo "Model Dir is empty: ${MODEL_PATH}"
fi

if [[ ! -f ${LABEL_PATH} ]]; then
    # wget -O ${LABEL_PATH} ${REMOTE_LABEL_PATH}
    # if [ $? -ne 0 ]; then
    #     echo "Download ${REMOTE_LABEL_PATH} to ${LABEL_PATH} failed."
    #     exit 1
    # fi
    echo "Label file is not exist: ${LABEL_PATH}"
fi

# generate config file with selected sinker and selected platform
pushd ${CURRENT_DIR}
    sed 's/__PLATFORM_PLACEHOLDER__/'"${1}"'/g' config_template.json | sed 's/__SINKER_PLACEHOLDER__/'"${2}"'.json/g' &> config.json
popd

mkdir -p output
# ${SAMPLES_ROOT}/bin/cns_launcher  \
#     --data_path ${SAMPLES_ROOT}/files.list_video \
#     --src_frame_rate 30 \
#     --config_fname ${CURRENT_DIR}/config.json \
#     --log_to_stderr=true

# gdb --args \
${SAMPLES_ROOT}/bin/cns_launcher  \
    --data_path ${CURRENT_DIR}/configs/test_${3}.txt \
    --src_frame_rate 30 \
    --config_fname ${CURRENT_DIR}/config.json \
    --log_to_stderr=true \
    --loop false \
    --udp_port 6868
    # --data_path ${SAMPLES_ROOT}/cns_launcher/det/tools/file_list_bdd100k_val.txt \files.list_image
