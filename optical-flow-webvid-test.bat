SET TF_ENABLE_ONEDNN_OPTS=0

video2dataset --tmp_dir="./tmp/" --url_list="data/webvid10m-test/{00000..00000}.tar" --input_format="webdataset" --output-format="webdataset" --output_folder="data/webvid10m-test-optical_flow" --stage "optical_flow" --encode_formats "{\"optical_flow\": \"npy\"}" --config "optical_flow"