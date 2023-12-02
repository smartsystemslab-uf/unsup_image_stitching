#!/usr/bin/env bash

# Function to configure the environment variables based on the dataset
configure_dataset() {
  case "$1" in
    "SmartSysLab")
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/SmartSysLab/cam0/
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/SmartSysLab/cam0/
        SCALE_FACTOR=3
      ;;
    "Malibu")
        # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Malibu/Calibration/calib_img
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Malibu/4
        INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Malibu/Input/{camID:05d}/{imgID:05d}.jpg
        CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Malibu/Input/{camID:05d}/{imgID:05d}.jpg
        NBC=5 # Number of camera
        NBI=400 # Maximum number of image to use for calibration
        NBIS=10 # Maximum number of image to use for image stitching
        SCALE_FACTOR=4
      ;;
    "Street")
        # Data Test 3
        # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Street/Calibration
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Street/4
        INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Street/Input/{camID:05d}/{imgID:05d}.jpg
        CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Street/Input/{camID:05d}/{imgID:05d}.jpg
        NBC=14 # Number of camera
        NBI=486 # Maximum number of image to use for calibration
        NBIS=10 # Maximum number of image to use for image stitching
        SCALE_FACTOR=4
      ;;
    "Terrace")
        # Data Test 4
        # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Terrace/Calibration
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Terrace3/4
        CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Terrace/Input/{camID:05d}/{imgID:05d}.jpg
        INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Terrace/Input/{camID:05d}/{imgID:05d}.jpg
        DFS="MCMI"
        NBC=14 # Number of camera
        NBIS=10 # Maximum number of image to use for image stitching
        NBI=430 # Maximum number of image to use for calibration
        SCALE_FACTOR=3
      ;;
    "Airplanes")
        # Data Test 5
        # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Airplanes/Calibration
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Airplanes2/4
        CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Airplanes/Input/{camID:05d}/{imgID:05d}.jpg
        INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Airplanes/Input/{camID:05d}/{imgID:05d}.jpg
        NBC=5 # Number of camera
        NBI=599 # Maximum number of image to use for calibration (162)
        NBIS=10 # Maximum number of image to use for image stitching
        SCALE_FACTOR=3
      ;;
    "Windmills")
        # Data Test 6 --> Not matching features for this dataset. (GAUSS_WINDOW_FACTOR 3, RANSAC_INLIER_THRES 2.5)
        # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Windmills/Calibration
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Windmills2/4
        INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Windmills/Input/{camID:05d}/{imgID:05d}.jpg
        CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Windmills/Input/{camID:05d}/{imgID:05d}.jpg
        NBC=5 # Number of camera
        NBI=300 # Maximun number of images to use for calibration
        NBIS=10 # Maximum number of image to use for image stitching
        SCALE_FACTOR=4
      ;;
    "Opera")
        # Data Test 7
        # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Opera/Calibration
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Opera/4
        CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Opera/Input/{camID:05d}/{imgID:05d}.jpg
        INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Opera/Input/{camID:05d}/{imgID:05d}.jpg
        NBC=5 # Number of camera
        NBI=450 # Maximum number of image to use for calibration
        NBIS=10 # Maximum number of image to use for image stitching
        SCALE_FACTOR=3
      ;;
    "CMU0")
        # Data Test 8
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/CMU0/4
        INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/Campus/CMU0/medium{imgID:02d}.jpg
        NBIS=37 # Maximum number of image to use for image stitching
        DFS="SCMI"
        SCALE_FACTOR=2
      ;;
    "CMU1")
        # Data Test 9
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/Campus/CMU1
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/CMU1/4
        DFS="IDIR"
        SCALE_FACTOR=2
      ;;
    "UFWEST4")
        # Data Test 10
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/UFCampus/UFWEST4
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/UFWEST4/4
        DFS="IDIR"
        SCALE_FACTOR=3
      ;;
    "zijing")
        # Data Test 11
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/zijing
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/zijing/4
        DFS="IDIR"
        SCALE_FACTOR=3
      ;;
    "flower")
        # Data Test 12
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/flower
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/flower/4
        DFS="IDIR"
        SCALE_FACTOR=1
      ;;
    "NSH")
        # Data Test 13
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/NSH
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/NSH/4
        DFS="IDIR"
        SCALE_FACTOR=3
      ;;
    "LAR230LAB")
        # Data Test 14
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/UFCampus/LAR230LAB
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/LAR230LAB/4
        DFS="IDIR"
        SCALE_FACTOR=3
      ;;
    "flower_artifacts")
        # Data Test 15
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/flower_artifacts
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/flower_artifacts/4
        DFS="IDIR"
        SCALE_FACTOR=1
      ;;
    "flower_low_resolution")
        # Data Test 16
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/flower_low_resolution
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/flower_low_resolution/4
        DFS="IDIR"
        SCALE_FACTOR=0.15
      ;;
    "flower_noise")
        # Data Test 17
        IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/flower_noise
        OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/flower_noise/4
        DFS="IDIR"
        SCALE_FACTOR=1
      ;;
    # Add additional datasets as needed
    *)
      echo "Unknown dataset: $1"
      return 1
      ;;
  esac
  return 0
}

# Check if a dataset is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

# Call the function with the dataset name
DATASET_NAME=$1
if ! configure_dataset "$DATASET_NAME"; then
  exit 1
fi

# Common environment setup
DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIB_DIR=$DIRECTORY/libs
METRIC=lpips

# Export library path for dependencies
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR/pano
# Model: is, eis, dis, ddis, rnis, distilled_rnis (not yet trained)
# New model: unrnis, unddis, unetis
# Run the stitching process with the configured variables
python main_stitching.py --imgdir $IMGDIR --outdir $OUTDIR --model unddis \
    --calib_pattern $CALIB_PATTERN --input_pattern $INPUT_PATTERN \
    -nbc $NBC -nbi $NBI -nbis $NBIS --scale_factor=$SCALE_FACTOR --dfs=$DFS \
    --files=$IMGFILES --metric=$METRIC
