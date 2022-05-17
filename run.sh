#!/bin/bash

python_file="./pipeline.py"

if [[ $# -eq 1 ]] || [[ $# -eq 2 ]]; then
  input_video=$1

	if [ -e $input_video ]; then
    python3 $python_file $input_video $mode
    excho "a"
	else 
      echo "Video does not exist"
      exit 1
	fi

  if [ $# -eq 2 ]; then
    mode=$2
    python3 $python_file $input_video $mode
  fi

   if [ $? -eq 0 ]; then
    echo "Successfully processed video"
  else
    echo "Failed to process video"
    exit 2
  fi

  IFS='/' read -ra input_parts <<< $input_video
  output_video="output_videos/${input_parts[-1]}"
  which vlc > /dev/null
  if [[ $? -eq 0 ]]; then
    vlc $output_video
  else
    echo "Output video is stored on: $output_video"
    echo "Use your favorite video player to play the video"
  fi

else
  echo "Usage: ./run.sh <input_video> [mode]"
	echo "Example: ./run.sh ./input.mp4"
	echo "Example: ./run.sh ./input.mp4 debug"
fi
