# Deep Learning Model for Image Stitching

This project is a deep learning-based approach to image stitching, inspired by and built upon the concepts from the [Image Super Resolution Repos](https://github.com/titu1994/Image-Super-Resolution).

## Overview
Image stitching is a technique in computer vision where multiple photographic images with overlapping fields of view are combined to produce a segmented panorama or high-resolution image. This repository contains a deep learning model that automates this process.

## Getting Started

### Prerequisites
To run this project, you will need:
- Python 3.x
- TensorFlow 2.x
- Additional Python libraries such as NumPy, scikit-learn (you can install these using `pip install -r requirements.txt` if you have a `requirements.txt` file)

### Data Preparation
Generate the training dataset from real images using the following script:

```bash
./prepare_data.sh
```

### Training the Model
Train the model with the following command:

```bash
python train.py --model UnDDStitch --batch_size=16 --nb_epochs=20 --load_weights 1 --save_model_img 0 --supervised 1
```

### Performing Inference
Use the `main_stitching.sh` script to stitch images:

```bash
./main_stitching.sh <dataset>
```

For example:

```bash
./main_stitching.sh UFWEST4
```

## Research Paper

This project is accompanied by our research paper titled "Semi-Supervised Image Stitching from Unstructured Camera Arrays" in Sensors. The paper presents a novel approach for stitching images from large unstructured camera sets, enhancing time efficiency and accuracy.

### Citation
If this project assists in your research, please consider citing our paper:

```bibtex
@Article{s23239481,
AUTHOR = {Nghonda Tchinda, Erman and Panoff, Maximillian Kealoha and Tchuinkou Kwadjo, Danielle and Bobda, Christophe},
TITLE = {Semi-Supervised Image Stitching from Unstructured Camera Arrays},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {23},
ARTICLE-NUMBER = {9481},
URL = {https://www.mdpi.com/1424-8220/23/23/9481},
ISSN = {1424-8220},
ABSTRACT = {Image stitching involves combining multiple images of the same scene captured from different viewpoints into a single image with an expanded field of view...},
DOI = {10.3390/s23239481}
}
```

For more details, please visit the [paper's webpage](https://www.mdpi.com/1424-8220/23/23/9481).

## Contributing
We welcome contributions to this project. Whether it's reporting a bug, proposing an enhancement, or submitting a pull request, your input is valuable. Please refer to our contribution guidelines for more information on how to contribute.

## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.