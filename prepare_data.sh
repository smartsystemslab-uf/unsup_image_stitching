#!/usr/bin/env bash

DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIB_DIR=$DIRECTORY/libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR/pano && python prepare_data.py --supervised 1