<div align="center">
<h1>FCViT MT NTNU</h1>
<h3>Solving Jigsaw Puzzles by Predicting Fragment’s Coordinate Based on Vision Transformer</h3>

Garam Kim<sup>a</sup>, Hyeonseong Cho<sup>a</sup> \*, Hyoungsik Nam<sup>a</sup> \*

<sup>a</sup> Kyung Hee University, Republic of Korea

(\*) Corresponding Authors

ESWA 2025
</div>
<br>



## FCViT: Fragment’s Coordinate prediction Vision Transformer
* This repository contains PyTorch training code and evaluation code for FCViT.
* Architecture of FCViT: 
* ![architecture](https://github.com/user-attachments/assets/87ac17a0-2590-4bdc-bb03-a8f1937add0c)
* For details see [Solving Jigsaw Puzzles by Predicting Fragment’s Coordinate Based on Vision Transformer](https://www.sciencedirect.com/science/article/abs/pii/S0957417425003987) by Garam Kim, Hyeonseong Cho and Hyoungsik Nam.
* If you use this code for a paper please cite:
* ```
  @article{kim2025solving,
  title={Solving jigsaw puzzles by predicting fragment’s coordinate based on vision transformer},
  author={Kim, Garam and Cho, Hyeonseong and Nam, Hyoungsik},
  journal={Expert Systems with Applications},
  volume={272},
  pages={126776},
  year={2025},
  publisher={Elsevier}
  }
  ```
<br>



### Catalog
- [x] Overview of paper
- [x] Usage
- [x] Data preparation
- [x] Evaluation code
- [x] Training code
- [x] License
<br>



### Overview of paper
* [`Overview in English`](OVERVIEW_ENG.md)
* [`Overview in Korean`](OVERVIEW_KOR.md)
<br>


### Usage
* First, clone the repository locally:
* ```
  git clone https://github.com/HiMyNameIsDavidKim/fcvit.git
  ```
* Then, install PyTorch and torchvision and [`timm==0.4.12`](https://github.com/rwightman/pytorch-image-models):
* ```
  conda install -c pytorch pytorch torchvision
  pip install timm==0.4.12
  ```
<br>



### Data preparation
* Download ImageNet train and val images from http://image-net.org/.
* The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:
* ```
  /path/to/imagenet/
    train/
      class1/
        img1.jpeg
      class2/
        img2.jpeg
    val/
      class1/
        img3.jpeg
      class2/
        img4.jpeg
  ```
<br>



### Evaluation code
* To evaluate a FCViT-base on ImageNet val with a GPU:
* ```
  python main_eval.py \
  --eval \
  --backbone vit_base_patch16_224 \
  --size_puzzle 225 \
  --size_fragment 75 \
  --num_fragment 9 \
  --batch_size 64 \
  --resume FCViT_base_3x3_ep100_lr3e-05_b64.pt \
  --data_path ${IMAGENET_DIR}
  ```
<br>



### Training code
* To train FCViT-base on ImageNet on a GPU for 100 epochs run:
* ```
  python main_train.py \
  --backbone vit_base_patch16_224 \
  --size_puzzle 225 \
  --size_fragment 75 \
  --num_fragment 9 \
  --lr 3e-05 \
  --epochs 100 \
  --weight_decay 0.05 \
  --batch_size 64 \
  --data_path ${IMAGENET_DIR} \
  --output_dir ${SAVE_DIR}
  ```
<br>



### Acknowledgments
* Our codebase is mainly based on [JigsawCFN](https://arxiv.org/abs/1603.09246), [MAE](https://arxiv.org/abs/2111.06377), [ViT](https://arxiv.org/abs/2010.11929) and [timm](https://github.com/rwightman/pytorch-image-models).
* This work was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Science, ICT & Future Planning (NRF-2019R1F1A1061114) and the Brain Korea 21 Four Program in 2022.
<br>
