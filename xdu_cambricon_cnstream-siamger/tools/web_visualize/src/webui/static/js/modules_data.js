window.MODULES = [{
class_name: "cnstream::DataSource",
name: "source",
parallelism: 0,
custom_params: {
    output_type: "mlu",
    decoder_type: "mlu",
    reuse_cndec_buf: true,
    input_buf_number: "",
    output_buf_number: "",
    interval: "",
    apply_stride_align_for_scaler: "",
    device_id: 0,
  },
  desc: "DataSource is a module for handling input data (videos or images). Feed data to codec and send decoded data to the next module if there is one.",
  label: "DataSource",
},
{
  class_name: "cnstream::Inferencer",
  name: "detector",
  parallelism: 1,
  max_input_queue_size: 20,
  custom_params: {
    model_path: "../../models/yolov3_b4c4_argb_mlu270.cambricon",
    func_name: "subnet0",
    use_scaler: "false",
    preproc_name: "",
    postproc_name: "PostprocSsd",
    batching_timeout: 100,
    threshold: 0.5,
    data_order: "",
    infer_interval: "",
    object_infer: "",
    obj_filter_name: "",
    keep_aspect_ratio: "",
    dump_resized_image_dir: "",
    model_input_pixel_format: "",
    mem_on_mlu_for_postproc: "",
    saving_infer_input: "",
    pad_method: "",
    device_id: 0,
  },
  desc: "Inferencer is a module for running offline model inference, as well as preprocessing and postprocessing.",
  label: "Inferencer",
},
{
  class_name: "cnstream::Inferencer2",
  name: "inferencer2",
  parallelism: 1,
  max_input_queue_size: 20,
  custom_params: {
    model_path: "../../models/yolov3_b4c4_argb_mlu270.cambricon",
    func_name: "subnet0",
    model_data: "",
    model_graph: "",
    preproc_name: "cncv",
    postproc_name: "VideoPostprocYolov3",
    engine_num: "",
    batching_timeout: 100,
    batch_strategy: "",
    priority: "",
    threshold: 0.5,
    data_order: "",
    show_stats: "",
    object_infer: "",
    obj_filter_name: "",
    keep_aspect_ratio: "true",
    mean: "",
    std: "",
    normalize: "",
    model_input_pixel_format: "ARGB32",
    device_id: 0,
  },
  desc: "Inferencer2 is a module for running offline model inference, as well as preprocessing and postprocessing.",
  label: "Inferencer2",
},
{
  class_name: "cnstream::Osd",
  name: "osd",
  parallelism: 4,
  max_input_queue_size: 20,
  custom_params: {
    label_path: "../../models/label_map_coco.txt",
    font_path: "",
    label_size: "",
    text_scale: "",
    text_thickness: "",
    box_thickness: "",
    secondary_label_path: "",
    attr_keys: "",
    logo: "",
  },
  desc: "Osd is a module for drawing objects on image. Output image is BGR24 format.",
  label: "Osd",
},
{
  class_name:"cnstream::Tracker",
  name:"tracker",
  parallelism: 16,
  max_input_queue_size: 20,
  custom_params: {
    model_path: "../../models/feature_extract_for_tracker_b4c4_argb_mlu270.cambricon",
    func_name: "subnet0",
    track_name: "",
    max_cosine_distance: "",
    device_id: 0,
  },
  desc: "Tracker is a module for realtime tracking.",
  label: "Tracker",
},
{
  class_name: "cnstream::RtspSink",
  name: "rtsp_sink",
  parallelism: 2,
  max_input_queue_size: 20,
  custom_params: {
    http_port: 8080,
    udp_port: 9554,
    frame_rate:25,
    gop_size: 30,
    kbit_rate: 512,
    preproc_type: "cpu",
    encoder_type: "mlu",
    color_mode: "bgr",
    view_mode: "single",
    view_rows: "",
    view_cols: "",
    dst_width: "",
    dst_height: "",
    device_id: 0,
  },
  desc: "RtspSink is a module to deliver stream by RTSP protocol.",
  label: "RtspSink",
},
{
  class_name:"cnstream::Encode",
  name:"encoder",
  parallelism: 2,
  max_input_queue_size: 20,
  custom_params: {
    preproc_type: "cpu",
    encoder_type: "mlu",
    codec_type: "h264",
    dst_width: 1920,
    dst_height: 1080,
    frame_rate: 25,
    kbit_rate: 512,
    gop_size: 30,
    use_ffmpeg: "false",
    output_dir: "./output",
    device_id: 0,
  },
  desc: "Encode is a module to encode videos or images.",
  label: "Encode",
},
{
  class_name:"cnstream::Displayer",
  name:"displayer",
  parallelism: 2,
  max_input_queue_size: 20,
  custom_params: {
    "window-width": 500,
    "window-height": 500,
    "refresh-rate": 30,
    "max-channels": 64,
    "full-screen": "true",
    show: "true",
  },
  desc: "Displayer is a module for displaying video.",
  label: "Displayer",
},
];

