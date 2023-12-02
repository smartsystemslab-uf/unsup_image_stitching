import logging

# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging_format = "%(asctime)s: %(levelname)s:%(message)s"
dataset_folder = "/mnt/data/enghonda/deepstitch-dataset"
image_folder = f"{dataset_folder}/training_data"
un_image_folder = f"{dataset_folder}/un_training_data"

project_dir = "."  # "/home/erman/projects/Image-Stitching-NN"
project_setting_dir = project_dir + "/settings"
libs_dir = project_dir + "/libs"
log_dir = project_dir + "/logs"
pano_libs_dir = libs_dir + "/pano"
pano_config_file = project_setting_dir + "/config.cfg"

# Data preparation settings
patch_size = 512
patch_step = 128
un_patch_size = 256  # Default: 512 Unsupervised patch size
un_patch_step = 96  # Default: 256 Unsupervised step size
sandfall_layer = -5

config_img_input = project_setting_dir + "/config_input_file.json"
config_img_output = image_folder + "/config_output_file.json"
un_config_img_output = un_image_folder + "/config_output_file.json"


class Colors:
    """ANSI color codes"""

    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    # cancel SGR codes if we don't write to a terminal
    if not __import__("sys").stdout.isatty():
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        # set Windows console in VT mode
        if __import__("platform").system() == "Windows":
            kernel32 = __import__("ctypes").windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            del kernel32


def PRINT_INFO(msg):
    print(f"{Colors.GREEN}{msg}{Colors.END}")


def PRINT_WARNING(msg):
    print(f"{Colors.YELLOW}{msg}{Colors.END}")


def PRINT_ERROR(msg):
    print(f"{Colors.YELLOW}{msg}{Colors.END}")
