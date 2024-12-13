<div align="center">
<h1>FCViT </h1>
<h3>Solving Jigsaw Puzzles by Predicting Fragment’s Coordinate Based on Vision Transformer</h3>

Garam Kim<sup>a</sup>, Hyeonseong Cho<sup>a</sup> \*, Hyoungsik Nam<sup>a</sup> \*

<sup>a</sup> Department of Information Display, Kyung Hee University

(\*) Corresponding Authors

ESWA 2025
</div>
<br>



## FCViT: Fragment’s Coordinate prediction Vision Transformer
* This repository contains PyTorch training code and evaluation code for FCViT.
* Architecture of FCViT: 
* [Image private]
* For details see [Solving Jigsaw Puzzles by Predicting Fragment’s Coordinate Based on Vision Transformer (private)]() by Garam Kim, Hyeonseong Cho and Hyoungsik Nam.
* If you use this code for a paper please cite:
* ```
  @article{kim25,
    title={Solving Jigsaw Puzzles by Predicting Fragment’s Coordinate Based on Vision Transformer},
    author={Kim, Garam and Cho, Hyeonseong and Nam, Hyoungsik},
    journal={Expert Systems with Applications},
    note = {In Review},
  }
  ```
<br>



### Catalog
- [x] Overview of paper
- [x] Usage
- [x] Data preparation
- [x] Evaluation code
- [x] Training code
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
* (추가할거 체크)
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
  python main.py \
  --eval \
  --model vit_base_patch16_224 \
  --batch_size 16 \
  --data_path ${IMAGENET_DIR}
  ```
<br>



### Training code
* 
<br>



### License
* This project is currently private and not yet available to the public.
<br>

