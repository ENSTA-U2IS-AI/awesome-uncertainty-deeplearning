# awesome-uncertainty-deeplearning

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

This repo is a collection of AWESOME papers/codes/blogs about Uncertainty and Deep learning, including papers, code, etc. Feel free to star and fork.

# Contents
<!-- - [awesome-domain-adaptation](#awesome-domain-adaptation) -->
- [Contents](#contents)
- [Papers](#papers)
  - [Survey](#survey)
  - [Theory](#theory)
  - [Ensemble/Bayesian-Methods](#ensemblebayesian-methods)
  - [Sampling/Dropout-based-Methods](#samplingdropout-based-Methods)
  - [Learning-loss/Auxiliary-network-Methods](#learning-lossauxiliary-network-methods)
  - [Data-augmentation/Generation-based-Methods](#data-augmentationgeneration-based-methods)
  - [Calibration](#Calibration)
  - [Prior-networks/Evidential-deep-learning](#Prior-networksevidential-deep-learning)
  - [Deterministic-Uncertainty-Methods](#Deterministic-Uncertainty-Methods)
  - [Quantile-Regression/Predicted-Intervals](#quantile-regressionpredicted-intervals)
  - [Applications](#Applications)
    - [Classification](#Classification)
    - [Semantic-Segmentation](#Semantic-Segmentation)
    - [Regression](#Regression)
    - [Annomaly-detection](#Annomaly-detection)
    - [Out of Distribution Dectection](#Out-of-Distribution-Dectection)
- [Datasets](#Datasets)
- [Benchmarks](#Benchmarks)
- [Library](#Library)
- [Lectures and Tutorials](#Lectures-and-tutorials)
- [Other Resources](#Other-resources)


# Papers
## Survey
**Arxiv**
- Ensemble deep learning: A review. [[6 Apr 2021]](https://arxiv.org/abs/2104.02395)
- 22
- 33


**Conference**
- 11
- 22
- A Comparison of Uncertainty Estimation Approaches in Deep Learning Components for Autonomous Vehicle Applications[[AISafety2020 Workshop]](https://arxiv.org/abs/2006.15172)

**Journal**
- A review of uncertainty quantification in deep learning: Techniques, applications and challenges [[Information Fusion]](https://www.sciencedirect.com/science/article/pii/S1566253521001081)
- 22
- 33

## Theory
**Arxiv**
- 11
- 22
- 33


**Conference**
- 11
- 22
- Evidential Deep Learning to Quantify Classification Uncertainty [[NIPS2018]](https://arxiv.org/abs/1806.01768) [[Pytorch]](https://github.com/dougbrion/pytorch-classification-uncertainty)
- To Trust Or Not To Trust A Classifier [[NIPS2018]](https://arxiv.org/abs/1805.11783) 

**Journal**
- 11
- 22
- 33

## Ensemble/Bayesian-Methods
**Arxiv**
- Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification [[arxiv2020]](https://arxiv.org/abs/2012.02818)
- Deep Ensembles: A Loss Landscape Perspective [[arxiv2019]](https://arxiv.org/abs/1912.02757)

**Conference**
- 11
- 33
- 22
- A General Framework for Uncertainty Estimation in Deep Learning [[ICRA2020]](https://arxiv.org/pdf/1907.06890.pdf)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](https://github.com/ezjong/lightprobnets) [[Pytorch]](https://github.com/ezjong/lightprobnets)
- Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning [[ICML2018]](http://proceedings.mlr.press/v80/depeweg18a.html)
- High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach [[ICML2018]](https://arxiv.org/abs/1802.07167) [[Tensorflow]](https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals)
- Uncertainty estimates and multi-hypotheses networks for optical flow [[ECCV2018]](https://arxiv.org/abs/1802.07095) [[Tensorflow]](https://github.com/lmb-freiburg/netdef_models)
- Simple and scalable predictive uncertainty estimation using deep ensembles [[NIPS2017]](https://arxiv.org/abs/1612.01474)

**Journal**
- One Versus all for deep Neural Network for uncertaInty (OVNNI) quantification [[IEEE Access2021]](https://arxiv.org/abs/2006.00954)
- Bayesian modeling of uncertainty in low-level vision [[IJCV1990]](https://link.springer.com/article/10.1007%2FBF00126502)

## Sampling/Dropout-based-Methods
**Arxiv**
- SoftDropConnect (SDC) – Effective and Efficient Quantification of the Network Uncertainty in Deep MR Image Analysis [[20 Jan 2022]](https://arxiv.org/abs/2201.08418)
- 22
- 33


**Conference**
- Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate [[AAAI2022]](https://arxiv.org/abs/1910.04858v3)
- Dropout Sampling for Robust Object Detection in Open-Set Conditions [[ICRA2018]](https://arxiv.org/abs/1710.06677)
- Concrete Dropout [[NIPS2017]](https://arxiv.org/abs/1705.07832)
- Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning [[ICML2016]](https://arxiv.org/abs/1506.02142)

**Journal**
- 11
- 22
- 33

## Learning-loss/Auxiliary-network-Methods
**Arxiv**
- 11
- 22
- Learning Confidence for Out-of-Distribution Detection in Neural Networks[[arxiv2018]](https://arxiv.org/abs/1802.04865)

**Conference**
- SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](https://arxiv.org/abs/2104.02395) [[Pytorch]](https://github.com/xuanlongORZ/SLURP_uncertainty_estimate) 
- Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](https://arxiv.org/abs/2002.05582)
- On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](https://arxiv.org/abs/2005.06209) [[Pytorch]](https://github.com/mattpoggi/mono-uncertainty)
- Addressing failure prediction by learning model confidence [[NeurIPS2019]](https://papers.nips.cc/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf)[[Pytorch]](https://github.com/valeoai/ConfidNet)
- Learning loss for active learning [[CVPR2019]](https://arxiv.org/abs/1905.03677) [[Pytorch]](https://github.com/Mephisto405/Learning-Loss-for-Active-Learning) (unofficial codes)
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?[[NIPS2017]](https://arxiv.org/abs/1703.04977) 
- Estimating the Mean and Variance of the Target Probability Distribution [[(ICNN94)]](https://ieeexplore.ieee.org/document/374138)

**Journal**
- Confidence Estimation via Auxiliary Models [[TPAMI2021]](https://arxiv.org/abs/2012.06508)
- 22
- 33


## Data-augmentation/Generation-based-methods
**Arxiv**
- 11
- 22
- 33


**Conference**
- Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [[ECCV2020]](https://arxiv.org/abs/2003.08440)[[Pytorch]](https://github.com/YingdaXia/SynthCP) (not sure)
- Detecting the Unexpected via Image Resynthesis [[ICCV2019]](https://arxiv.org/abs/1904.07595)[[Pytorch]](https://github.com/cvlab-epfl/detecting-the-unexpected) (not sure)
- On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks [[NIPS2019]](https://arxiv.org/abs/1905.11001)

**Journal**
- 11
- 22
- 33


## Calibration
**Arxiv**
- The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration [[arxiv2021]](https://arxiv.org/abs/2111.15430)[[Pytorch]](https://github.com/by-liu/mbls)
- Evaluating and Calibrating Uncertainty Prediction in Regression Tasks [[arxiv2020]](https://arxiv.org/abs/1905.11659)
- 22
- 33


**Conference**
- Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning [[MIDL2020]](http://proceedings.mlr.press/v121/laves20a.html) [[Pytorch]](https://github.com/mlaves/well-calibrated-regression-uncertainty)
- Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision [[CVPRW2020]](https://arxiv.org/abs/1906.01620) [[Pytorch]](https://github.com/fregu856/evaluating_bdl)
- Measuring Calibration in Deep Learning [[CVPRW2019]](https://arxiv.org/abs/1904.01685)
- Accurate Uncertainties for Deep Learning Using Calibrated Regression [[ICML2018]](https://arxiv.org/abs/1807.00263)
- On calibration of modern neural networks. [[ICML2017]](https://arxiv.org/abs/1706.04599)

**Journal**
- Calibrated Prediction Intervals for Neural Network Regressors [[IEEE Access 2018]](https://arxiv.org/abs/1803.09546)[[Python]](https://github.com/cruvadom/Prediction_Intervals)
- 22
- 33

## Prior-networks/Evidential-deep-learning
**Arxiv**
- 11
- 22
- 33


**Conference**
- Improving Evidential Deep Learning via Multi-task Learning [[AAAI2022]](https://arxiv.org/abs/2112.09368)
- Deep Evidential Regression [[NIPS2020]](https://arxiv.org/abs/1910.02600) [[Tensorflow]](https://github.com/aamini/evidential-deep-learning)
- Evidential Deep Learning to Quantify Classification Uncertainty [[NIPS2018]](https://arxiv.org/abs/1806.01768) [[Pytorch]](https://github.com/dougbrion/pytorch-classification-uncertainty)

**Journal**
- 11
- 22
- 33

## Deterministic-Uncertainty-Methods

**Arxiv**
- 11
- 22
- 33


**Conference**
- 11
- 22
- Single-Model Uncertainties for Deep Learning [[NIPS2019]](https://arxiv.org/abs/1811.00908) [[Pytorch]](https://github.com/facebookresearch/SingleModelUncertainty/)

**Journal**
- 11
- 22
- 33

## Quantile-Regression/Predicted-Intervals

**Arxiv**
- 11
- 22
- 33


**Conference**
- Prediction Intervals: Split Normal Mixture from Quality-Driven Deep Ensembles [[UAI2020]](http://proceedings.mlr.press/v124/saleh-salem20a.html) [[Pytorch]](https://github.com/tarik/pi-snm-qde)
- Single-Model Uncertainties for Deep Learning [[NIPS2019]](https://arxiv.org/abs/1811.00908) [[Pytorch]](https://github.com/facebookresearch/SingleModelUncertainty/)
- High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach [[ICML2018]](https://arxiv.org/abs/1802.07167) [[Tensorflow]](https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals)

**Journal**
- Exploring uncertainty in regression neural networks for construction of prediction intervals [[Neurocomputing2022]](https://www.sciencedirect.com/science/article/abs/pii/S0925231222001102)
- 22
- 33

## Applications


### Classification
**Arxiv**
- 11
- 22
- 33


**Conference**
- 11
- 22
- Lightweight Probabilistic Deep Networks [[CVPR2018]](https://arxiv.org/abs/1805.11327)[[Pytorch]](https://github.com/ezjong/lightprobnets)

**Journal**
- 11
- 22
- 33


### Semantic-Segmentation
**Arxiv**
- 11
- 22
- Evaluating Bayesian Deep Learning Methods for Semantic Segmentation [[arxiv2018]](https://arxiv.org/abs/1811.12709)


**Conference**
- 11
- 22
- 33
- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/html/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.html)
- Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation [[MICCAI2019]](https://arxiv.org/abs/1806.05034)[[Pytorch]](https://github.com/yulequan/UA-MT)
- A Probabilistic U-Net for Segmentation of Ambiguous Images [[NIPS2018]](https://arxiv.org/abs/1806.05034) [[Pytorch]](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch)
- Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding [[BMVC2017]](https://arxiv.org/abs/1511.02680)

**Journal**
- 11
- 22
- 33

### Regression
**Arxiv**
- On Monocular Depth Estimation and Uncertainty Quantification using Classification Approaches for Regression [[arxiv2022]](https://arxiv.org/abs/2202.12369)
- Evaluating and Calibrating Uncertainty Prediction in Regression Tasks [[arxiv2020]](https://arxiv.org/abs/1905.11659)

**Conference**
- Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate [[AAAI2022]](https://arxiv.org/abs/1910.04858v3)
- SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](https://arxiv.org/abs/2104.02395) [[Pytorch]](https://github.com/xuanlongORZ/SLURP_uncertainty_estimate) 
- Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](https://arxiv.org/abs/2002.05582)
- Deep Evidential Regression [[NIPS2020]](https://arxiv.org/abs/1910.02600) [[Tensorflow]](https://github.com/aamini/evidential-deep-learning)
- Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning [[MIDL2020]](http://proceedings.mlr.press/v121/laves20a.html) [[Pytorch]](https://github.com/mlaves/well-calibrated-regression-uncertainty)
- On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](https://arxiv.org/abs/2005.06209) [[Pytorch]](https://github.com/mattpoggi/mono-uncertainty)
- Fast Uncertainty Estimation for Deep Learning Based Optical Flow [[IROS2020]](https://authors.library.caltech.edu/104758/)
- Inferring Distributions Over Depth from a Single Image [[IROS2019]](https://arxiv.org/abs/1912.06268) [[Tensorflow]](https://github.com/gengshan-y/monodepth-uncertainty)
- Multi-Task Learning based on Separable Formulation of Depth Estimation and its Uncertainty [[CVPRW]](https://openaccess.thecvf.com/content_CVPRW_2019/html/Uncertainty_and_Robustness_in_Deep_Visual_Learning/Asai_Multi-Task_Learning_based_on_Separable_Formulation_of_Depth_Estimation_and_CVPRW_2019_paper.html)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](https://arxiv.org/abs/1805.11327)[[Pytorch]](https://github.com/ezjong/lightprobnets)
- Uncertainty estimates and multi-hypotheses networks for optical flow [[ECCV2018]](https://arxiv.org/abs/1802.07095) [[Tensorflow]](https://github.com/lmb-freiburg/netdef_models)
- Accurate Uncertainties for Deep Learning Using Calibrated Regression [[ICML2018]](https://arxiv.org/abs/1807.00263)

**Journal**
- Exploring uncertainty in regression neural networks for construction of prediction intervals [[Neurocomputing2022]](https://www.sciencedirect.com/science/article/abs/pii/S0925231222001102)
- Calibrated Prediction Intervals for Neural Network Regressors [[IEEE Access 2018]](https://arxiv.org/abs/1803.09546)[[Python]](https://github.com/cruvadom/Prediction_Intervals)
- 22

### Annomaly-detection
**Arxiv**
- 11
- 22
- 33


**Conference**
- 11
- Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [[ECCV2020]](https://arxiv.org/abs/2003.08440)[[Pytorch]](https://github.com/YingdaXia/SynthCP) (not sure)
- Detecting the Unexpected via Image Resynthesis [[ICCV2019]](https://arxiv.org/abs/1904.07595)[[Pytorch]](https://github.com/cvlab-epfl/detecting-the-unexpected) (not sure)

**Journal**
- 11
- 22
- 33

### Out-of-Distribution-Dectection

**Arxiv**
- 11
- 22
- 33


**Conference**
- 11
- 22
- 33

**Journal**
- One Versus all for deep Neural Network for uncertaInty (OVNNI) quantification [[IEEE Access2021]](https://arxiv.org/abs/2006.00954)
- 22
- 33

# Datasets
- MUAD: Multiple Uncertainties for Autonomous Driving benchmark for multiple uncertainty types and tasks [[arxiv2022]](https://arxiv.org/abs/2203.01437)

# Benchmarks
- Uncertainty Baselines: Benchmarks for Uncertainty & Robustness in Deep Learning [[arxiv2021]](https://arxiv.org/abs/2106.04015)[[Tensorflow]](https://github.com/google/uncertainty-baselines)

# Library
- Uncertainty Toolbox [[github]](https://github.com/uncertainty-toolbox)

# Lectures-and-tutorials
- Yarin Gal: BAYESIAN DEEP LEARNING 101 [[website]](http://www.cs.ox.ac.uk/people/yarin.gal/website/bdl101/)
- MIT 6.S191: Evidential Deep Learning and Uncertainty (2021) [[Youtube]](https://www.youtube.com/watch?v=toTcf7tZK8c)

# Other-resources
