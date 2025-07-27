# FCViT
* Solving Jigsaw Puzzles with Vision Transformer-Based Coordinate Prediction
* ![fcvit architecture](https://github.com/user-attachments/assets/87ac17a0-2590-4bdc-bb03-a8f1937add0c)
* ImageNet 3x3 Puzzle Visualization
* ![puzzle visual](https://github.com/user-attachments/assets/8239f58b-772f-4676-8e77-c9fd514a82d9)
<br>

## Contents
### 👨‍🏫 Overview
### ✅ Define the problem
### 💡 Set a hypothesis
### 🔬 Experiments and validations
### 📊 Conclusion
<br>



## 👨‍🏫 Overview
* __Overview__: 
  * Jigsaw puzzles involve reassembling small fragments to create a complete image, requiring cognitive skills such as pattern recognition and spatial reasoning. Solving a jigsaw puzzle may seem simple to humans, but it's a complex challenge for computer vision models. Surprisingly, the current state of computer vision models is such that they are unable to solve even easy puzzles with as few as nine fragments. 
  * We want to solve these problems by applying two ideas: first, most existing models use classification algorithms, which we want to change to regression algorithms, and second, we want to solve the problem by utilizing Vision Transformers (ViT) as encoders instead of CNNs.
* __Duration/People__: 2023.02.01 ~ 2024.12.31 (2 years) / 3 people
* __Primary Role__: 
  * Fixes issues with classification algorithms with regression algorithms
  * Modify the architecture of the deep learning model ViT (Vision Transformer)
  * Designed data processing and learning strategies for self-supervised learning
  * Re-implement models from related papers with PyTorch
  * Visualize learning content and learning process
* __Lessons & Learns__: 
  * Think creatively to solve speed and performance problems in deep learning models
  * Improve the ability to read and understand papers and implement them in code
  * Understand how to work with large datasets (ImageNet)
  * Understand how to train models on Linux server GPU environments
  * Increase skills in Python programming
<br>



## ✅ Define the problem
![Jigsaw puzzle learning](https://github.com/user-attachments/assets/302663fc-07b0-438e-acb8-8791b5e00455)
* __What is jigsaw puzzle learning?__
  * Jigsaw puzzle learning is the reassembly of randomly mixed puzzle fragments.
  * It is a challenging task in the field of computer vision, requiring pattern recognition and spatial reasoning.
* __Why solve the problem of jigsaw puzzle learning?__
  * Deep learning models trained on jigsaw puzzles are good at `pattern recognition and spatial reasoning`.
  * `Object detection` is a prime example of how jigsaw puzzles can be trained and used in various fields.
  * Object detection is typically utilized in technologies such as `autonomous driving`.
  * Jigsaw puzzles are `self-supervised learning`.
  * Learning can be done from images alone, without human labeling.
  * Can be used as one of the methods of `massive pre-training`, a long-standing challenge in Computer Vision.
* __How to solve it?__
  * Method: Modify the architecture of the JigsawCFN model.
  * JigsawCFN was chosen because it is an end-to-end model with transfer learning in mind.
<br>



## 💡 Set a hypothesis
* __Improvement point 1: Classification algorithms__
  * ![class and reg](https://github.com/user-attachments/assets/5aa8f5bf-67ae-4c64-86a0-53fbff89cc5a)
  * Conventional approaches use a `classification algorithm` to predict the probabilities of all possible permutations.
  * The number of puzzle fragments and permutations are proportional, and `the number of permutations grows exponentially`.
  * (ex. 4 puzzle fragments have 4! (24) possible permutations, 9 puzzle fragments have 9! (362,880) possible permutations)
  * Therefore, the model size (= probabilities that the model has to compute) also grows exponentially.
  * The `classification algorithm` causes issues for the efficiency and scalability of jigsaw puzzle learning.
* __Hypothesis 1: Changing to a regression algorithm will solve the efficiency and scalability issues.__
  * The proposed learning strategy uses a `regression algorithm` to predict the horizontal and vertical coordinate values (h, v) of the puzzle fragments.
  * The number of puzzle fragments and coordinates are proportional, and `the number of coordinates increases linearly`.
  * (ex. 4 puzzle fragments only predict 4x2 (=8) coordinates, 9 puzzle fragments only predict 9x2 (=18) coordinates.)
  * Therefore, the model size (= coordinates to be calculated by the model) grows linearly.
  * `Regression algorithm` models are relatively small, `efficient`, and scalable to cover the increasing number of puzzle fragments. In addition, this method is characterized by its similarity to human behavior patterns.
* __Improvement point 2: CNN encoder__
  * ![CNN encoder](https://github.com/user-attachments/assets/abcc7319-ff45-4d2f-82eb-9f6483bc4417)
  * Most Conventional models use a `CNN encoder`.
  * CNNs have `local feature extraction limitations`.
  * Local features are limited to seeing the whole image and assembling the puzzle.
  * CNNs need to be changed to modern architectures to improve performance.
* __Hypothesis 2: Better performance can be achieved by replacing it with a ViT encoder.__
  * We can expect higher performance when using the `ViT encoder`.
  * ViT has good `global feature extraction ability`, which makes it suitable for puzzle problems.
  * The backbone is designed to be easily replaceable for ease of development.
<br>



## 🔬 Experiments and validations
* __FCViT Architecture__
  * ![fcvit architecture](https://github.com/user-attachments/assets/87ac17a0-2590-4bdc-bb03-a8f1937add0c)
  * Design data processing and learning strategies for self-supervised learning
    * Crop the input image into 9 puzzle fragments.
    * Assign a unique horizontal and vertical coordinate (h, v) to each puzzle fragment.
    * Randomly shuffle the 9 unique coordinates.
    * (The shuffled coordinates will be the label, which is the training answer).
    * Shuffle the puzzle fragments based on the shuffled coordinates.
    * (This shuffled image is input to the model.)
  * Design jigsaw puzzle model architecture with regression algorithm and ViT encoder
    * Input: A puzzle problem of size 224x224 (=mixed images).
    * Encoder: Extract features from the mixed image, ViT-16/B
    * Predictor: use features to predict unique coordinates, 3 MLPs, dim=1000, ReLU
    * Output: 9 predicted horizontal and vertical coordinates (h, v)
    * Loss function: Calculate the difference between the label and the prediction, SmoothL1 Loss
* __Dataset__
  * Training & Evaluation: ImageNet, JPwLEG
  * Evaluation: CIFAR10, iNaturalists19, MET
* __Experiment 1: Training and evaluation of the ImageNet dataset__
  * ![table 1](https://github.com/user-attachments/assets/66ab979a-f70b-49d2-9737-1b25c7d5819c)
  * Train and evaluate ImageNet 3×3 puzzles.
  * Improved from 83.3% to 90.6% when compared to the conventional SOTA model.
  * In conclusion, FCViT achieved SOTA.
* __Experiment 2: Evaluate encoder replacement__
  * ![table 5](https://github.com/user-attachments/assets/1053cd56-45dc-4694-931e-ac48ea1c9969)
  * FCCNN using only the regression algorithm learning strategy outperforms the conventional model.
  * This supports that `Hypothesis 1` is true.
  * FCViT with the additional use of the ViT encoder performs better than FCCNN.
  * This part supports that `Hypothesis 2` is true.
* __Experiment 3: Evaluate generalization performance__
  * ![table 3](https://github.com/user-attachments/assets/83b24f71-4afd-458e-b91b-551aa35bc6e3)
  * Only train on the ImageNet dataset and evaluate on the other datasets.
  * This evaluation evaluates whether the model can extract robustness features.
  * Robustness refers to generalized performance that is not specific to the trained dataset.
  * The robustness of the proposed model is reduced by 10% to 38%, while the existing model is reduced by 1% to 21%.
  * Therefore, FCViT can extract more robustness features compared to the conventional model.
* __Experiment 4: Evaluate computational efficiency__
  * ![table 4](https://github.com/user-attachments/assets/c62c7b1e-239e-4350-860d-628eb3c6878a)
  * Compare FCViT to the existing SOTA model, JPDVT.
  * Improved model size (131M -> 87M) and inference time (3.4 s -> 0.016 s).
  * FCViT is both a more efficient model and a more effective model.
  * JigsawCFN is a model that can only cover 1K permutations.
  * If this model covered 9! permutations, its size would be about 360 times larger.
  * FCViT, on the other hand, has the same size for both permutations of 1K and permutations of 9!
<br>



## 📊 Conclusion
* __Conclusion 1: Approaching jigsaw puzzle problems with coordinate prediction and ViT__ 
  * Conventional models approach the problem with classification algorithms.
  * FCViT approached the problem differently with a regression algorithm.
  * Improved model size by 34% and reduced inference time by 1 in 200.
  * Effects: FCViT enables us to tackle 4x4 and 5x5 puzzle problems that were previously impossible.
* __Conclusion 2: Achieving SOTA for jigsaw puzzle problems__
  * Improved accuracy by 7.3% on the ImageNet dataset, achieving a SOTA of 90.6%.
  * Learned more robustness representations than conventional models with 1% to 20% reduction in generalization performance.
  * Effects: Using jigsaw puzzle problems as pre-training, it can be utilized in various fields.
<br>
