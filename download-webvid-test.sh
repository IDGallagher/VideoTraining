#!/bin/bash

video2dataset --tmp_dir=".\tmp" --url_list=".\datasets\webvid10m\test.csv" --input_format="csv" --output-format="webdataset" --output_folder=".\data\webvid10m-test" --url_col="contentUrl" --caption_col="name" --save_additional_columns="[videoid]" --enable_wandb=True --config=default