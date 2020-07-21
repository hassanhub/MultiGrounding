# Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding
The original Tensorflow implementation for the paper [Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding](http://openaccess.thecvf.com/content_CVPR_2019/papers/Akbari_Multi-Level_Multimodal_Common_Semantic_Space_for_Image-Phrase_Grounding_CVPR_2019_paper.pdf) published in CVPR 2019.
![Sample1](http://hassanakbari.com/uploads/papers/CVPR19_Results.jpg)

## Download pre-processed data
Please first download and extract the data from the following links:

[Part 1 - 10 GB](https://hassanakbari.com/uploads/projects/multigrounding19/data/grounding_data.tar.gz.part_00),

[Part 2 - 10 GB](https://hassanakbari.com/uploads/projects/multigrounding19/data/grounding_data.tar.gz.part_01),

[Part 3 - 10 GB](https://hassanakbari.com/uploads/projects/multigrounding19/data/grounding_data.tar.gz.part_02),

[Part 4 - 10 GB](https://hassanakbari.com/uploads/projects/multigrounding19/data/grounding_data.tar.gz.part_03),

[Part 5 - 2.99 GB](https://hassanakbari.com/uploads/projects/multigrounding19/data/grounding_data.tar.gz.part_04)

## Download dependencies
The model depends on pre-trained models e.g. ELMo, VGG, and PNASNet. Please download these models from the following link and unpack them under `modules` folder.

[Modules - 6.3 GB](https://hassanakbari.com/uploads/projects/multigrounding19/data/modules.tar.gz)

## Training
To train a model, open and run all cells in each of the notebook files depending on the desired model.

## Dependencies
Python 3.6/3.7

Tensorflow 1.14.0

We also use the following packages:
- `tensorflow_hub`
- `opencv-python`

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
