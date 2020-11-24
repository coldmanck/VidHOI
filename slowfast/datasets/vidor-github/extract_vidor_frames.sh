#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

# Extract frames from videos.

OUT_DATA_DIR="frames"
if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

IN_DATA_DIR="video"
for folder in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  folder_name=${folder##*/}
  folder_name=${folder_name::-1}
  # echo $folder_name
  IN_DATA_DIR_INNER="video/${folder_name}"
  OUT_DATA_DIR="frames/${folder_name}"
  # if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  #   echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  #   mkdir -p ${OUT_DATA_DIR}
  # fi

  for video in $(ls -A1 -U ${IN_DATA_DIR_INNER}/*)
  do
    video_name=${video##*/}
    video_name=${video_name::-4}

    out_video_dir=${OUT_DATA_DIR}/${video_name}/
    mkdir -p "${out_video_dir}"

    out_name="${out_video_dir}/${video_name}_%06d.jpg"

    ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
  done
done