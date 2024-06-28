SET TF_ENABLE_ONEDNN_OPTS=0

.\video2dataset\.venv\Scripts\video2dataset.exe --tmp_dir=".\tmp" --url_list="./datasets/test.csv" --input_format="csv" --output-format="webdataset" --output_folder="./data/webvid10m-test" --url_col="contentUrl" --caption_col="name" --save_additional_columns="[videoid]" --enable_wandb=True --config=".\configs\default.yaml"

@REM .\video2dataset\.venv\Scripts\video2dataset.exe --tmp_dir=".\tmp" --url_list="./datasets/test.csv" --input_format="csv" --output-format="webdataset" --output_folder="s3://webvid10m-test/webvid10m-test" --url_col="contentUrl" --caption_col="name" --save_additional_columns="[videoid]" --enable_wandb=True --config=".\configs\default.yaml"