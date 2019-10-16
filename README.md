# Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding
Tensorflow implementation for the paper [Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding](http://openaccess.thecvf.com/content_CVPR_2019/papers/Akbari_Multi-Level_Multimodal_Common_Semantic_Space_for_Image-Phrase_Grounding_CVPR_2019_paper.pdf) published in CVPR 2019.
![Sample1](http://hassanakbari.com/uploads/papers/CVPR19_Results.jpg)

## Data Pre-Processing
You need to first follow the instructions in `./data/readme/` for each dataset and prepare the data. A sample code for creating data process instance can be found in `./data/data-process.ipynb`

## Pre-Trained Models
Please download pre-trained models from [here](https://www.dropbox.com/s/2tzxkpi86gdd60n/models.tar?dl=0) and unpack them in `./code/models/`.
Please note that this package also includes visual models (pre-trained on ImageNet) and ELMo model, which are necessary.

## Training
To train a model, simply run `./code/train.py` specifying the desired parameters.

## Dependencies
Python 3.6/3.7

Tensorflow 1.14.0

We also use the following packages, which could be installed by `pip install -r requirements.txt`:
- `tensorpack.dataflow`
- `tensorflow_hub`
- `opencv-python`

## ToDo List

- [x] Main Model
- [x] Parallel Data Pipeline
- [x] Distributed Training
- [x] Upload Pretrained Models
- [ ] Upload Pre-Processed Data
- [ ] Final Sanity Check

## Cite

If you found this work/code helpful or used this work in any capacity, please cite:
```
@inproceedings{akbari2019multi,
  title={Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding},
  author={Akbari, Hassan and Karaman, Svebor and Bhargava, Surabhi and Chen, Brian and Vondrick, Carl and Chang, Shih-Fu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12476--12486},
  year={2019}
}
```
