#!/bin/bash

python_file="./lanes_detector.py"

if [ $# -eq 1 ] || [ $# -eq 2 ]; then
  input_video_path=$1

	if [ -e $input_video_path ]; then
    python3 $python_file $input_video_path $mode
	else 
      echo "Video does not exist"
      exit 1
	fi

  if [ $# -eq 2 ]; then
    mode=$2
    python3 $python_file $input_video_path $mode
  fi

else
  echo "Usage: ./run.sh <input_video> [mode]"
	echo "Example: ./run.sh ./input.mp4"
	echo "Example: ./run.sh ./input.mp4 debug"
fi
