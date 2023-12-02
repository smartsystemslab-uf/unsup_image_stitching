from __future__ import division, print_function
import os
import time
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
try:
    import libpyopenpano as pano
except:
    ValueError(
        "Couldn't import 'libpyopenpano' library. You may need to use the shell "
        "script (*.sh files) to run this module or export LD_LIBRARY_PATH variable.\n"
        "    => Ex: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR && python prepare_stitching_data.py"
    )
# App Dir
DATA_DIR = "/home/lab/SSLAB_3D_CAM/stitching_pano_lib"
# if os.path.exists(f'{DATA_DIR}/libpyopenpano.so'):
#     print('File exists')
# else:
#     print('File does not exist')
# import ctypes
# pano = ctypes.cdll.LoadLibrary(os.path.join(DATA_DIR, 'libpyopenpano.so'))
# libpng = ctypes.CDLL(os.path.join(DATA_DIR, 'liblodepng.so'))
# libpano = ctypes.CDLL(os.path.join(DATA_DIR, 'libopenpano.so'))
# DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# LIB_DIR=$DIRECTORY/libs
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR 
# python3 panowrapper.py
def get_mat32f_list(path, exts = ["jpg", "jpeg", "png", "bmp"]):
    """Get a list of files with the given extension in the given directory."""
    
    file_list = get_file_list(path, exts)
    mat32f_list = [pano.create_mat32f(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)/255.0) for file in file_list]
    
    return mat32f_list
def get_file_list(path, exts = ["jpg", "jpeg", "png", "bmp"]):
    """Get a list of files with the given extension in the given directory."""
    file_list = []
    
    # exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
    for ext in exts:
        file_list.extend(
            [
                os.path.join(path, filename)
                for filename in os.listdir(path)
                if re.search(r"\." + ext + "$", filename, re.IGNORECASE)
            ]
        )
        
    print(file_list)
    return file_list
if __name__ == "__main__":    
    # help(pano.init_config)
    # Test Stitching
    pano_config_file = "config.cfg"
    pano.init_config(pano_config_file)
    pano.print_config()
    print(f" {pano_config_file:_^50}")
    # Get the list of images
    mdata = [
        {
            "input_dir": f"{DATA_DIR}/Campus/CMU0",
            "out_dir": f"{DATA_DIR}/Campus/CMU0/output",
        },
        {
            "input_dir": f"{DATA_DIR}/flower",
            "out_dir": f"{DATA_DIR}/flower/output",
        },
    ]
    id = 1
    out_dir = mdata[id]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    output_result = f"{out_dir}/{id:05d}.jpg"
    print(output_result)
    nb_stitched_img = 0
    stitcher = None
    
    # file_list = get_file_list(mdata[id]["input_dir"])
    file_list = get_mat32f_list(mdata[id]["input_dir"])
    print(file_list)
    stitcher = None
    try:
        # Instantiate the Stitcher
        stitcher = pano.Stitcher(file_list)
        # Stitch the images
        mat = stitcher.build()
        # Save the result
        pano.write_img(output_result, mat)        
    except:
        print(f"Error: Cannot stitch image [{id}] - [{output_result}]")
        exit(1)
    # Sleep for 1 second
    time.sleep(1)
    multi_band_blend = 10  # 0 is for linear blending
    # The function will stitch the next images without
    # re-initializing the stitcher and without recompting the features
    mat = stitcher.build_from_new_images_mat(file_list, multi_band_blend)
    # Save the result
    output_result = f"{out_dir}/{id:05d}.jpg"
    pano.write_img(output_result, mat)
    # Convert the image to a numpy array
    p = np.array(mat, copy=False)
    # Force the datatype of the RGB values to be within the range of 0 to 1
    data_clipped = np.clip(p, 0, 1)
    plt.imshow(data_clipped)
    plt.show()
