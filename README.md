# Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding
Tensorflow implementation for the paper [Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding](http://openaccess.thecvf.com/content_CVPR_2019/papers/Akbari_Multi-Level_Multimodal_Common_Semantic_Space_for_Image-Phrase_Grounding_CVPR_2019_paper.pdf) published in CVPR 2019.
![Sample1](http://hassanakbari.com/uploads/papers/CVPR19_Results.jpg)

### Data Pre-Processing
You need to first follow the instructions in `./data/readme/` for each dataset and prepare the data. A sample code for creating data process instance can be found in `./data/data-process.ipynb`

### Training
To train a model, simply run `./code/train.py` specifying the desired parameters.

### Dependencies
Python 3.6/3.7

Tensorflow 1.14.0

We also use the following packages, which could be installed by `pip install -r requirements.txt`:
- `tensorpack.dataflow`
- `tensorflow_hub`
- `opencv-python`

### ToDo List

- [x] Main Model
- [x] Parallel Data Pipeline
- [x] Distributed Training
- [] Upload Pretrained models
- [] Final Sanity Check
