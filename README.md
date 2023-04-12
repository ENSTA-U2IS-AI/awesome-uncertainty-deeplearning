# awesome-uncertainty-deeplearning

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo is a collection of AWESOME papers/codes/blogs about Uncertainty and Deep learning, including papers, code, etc. Feel free to star and fork.

if you think we missed a paper, or if you have any ideas for improvements, please send a message on the corresponding [GitHub discussion](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning/discussions). 

You may also an email at:
gianni.franchi at ensta-paris.fr with "[Awesome Uncertainty]" as subject (tell us where it was published and where, and send us a GitHub link and arxiv link if they are available).

## Table of Contents

- [awesome-uncertainty-deeplearning](#awesome-uncertainty-deeplearning)
  - [Table of Contents](#table-of-contents)
- [Papers](#papers)
  - [Surveys](#surveys)
  - [Theory](#theory)
  - [Bayesian-Methods](#bayesian-methods)
  - [Ensemble-Methods](#ensemble-methods)
  - [Sampling/Dropout-based-Methods](#samplingdropout-based-methods)
  - [Auxiliary-Methods/Learning-loss-distributions](#auxiliary-methodslearning-loss-distributions)
  - [Data-augmentation/Generation-based-methods](#data-augmentationgeneration-based-methods)
  - [Dirichlet-networks/Evidential-deep-learning](#dirichlet-networksevidential-deep-learning)
  - [Deterministic-Uncertainty-Methods](#deterministic-uncertainty-methods)
  - [Quantile-Regression/Predicted-Intervals](#quantile-regressionpredicted-intervals)
  - [Calibration](#calibration)
  - [Applications](#applications)
    - [Classification and Semantic-Segmentation](#classification-and-semantic-segmentation)
    - [Regression](#regression)
    - [Anomaly-detection and Out-of-Distribution-Detection](#anomaly-detection-and-out-of-distribution-detection)
    - [Object detection](#object-detection)
    - [Domain adaptation](#domain-adaptation)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Libraries](#libraries)
- [Lectures and tutorials](#lectures-and-tutorials)
- [Books](#books)
- [Other resources](#other-resources)

# Papers

## Surveys

**Conference**

- A Comparison of Uncertainty Estimation Approaches in Deep Learning Components for Autonomous Vehicle Applications[[AISafety Workshop2020]](https://arxiv.org/abs/2006.15172)

**Journal**

- Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods [[Machine Learning]](https://link.springer.com/article/10.1007/s10994-021-05946-3)
- Predictive inference with the jackknife+ [[The Annals of Statistic(2021)]](https://arxiv.org/abs/1905.02928)
- A review of uncertainty quantification in deep learning: Techniques, applications and challenges [[Information Fusion 2021]](https://www.sciencedirect.com/science/article/pii/S1566253521001081)
- A Survey on Uncertainty Estimation in Deep Learning Classification Systems from a Bayesian Perspective [[ACM2021]](https://dl.acm.org/doi/pdf/10.1145/3477140?casa_token=6fozCYTovlIAAAAA:t5vcjuXCMem1b8iFwaMG4o_YJHTe0wArLtoy9KCbL8Cow0aGEoxSiJans2Kzpm2FSKOg-4ZCDkBa)
- Uncertainty in big data analytics: survey, opportunities, and challenges [[Journal of Big Data2019]](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0206-3?cv=1)

**Arxiv**

- A Survey on Uncertainty Reasoning and Quantification for Decision Making: Belief Theory Meets Deep Learning [[arXiv2022]](https://arxiv.org/abs/2206.05675)
- Ensemble deep learning: A review [[arXiv2021]](https://arxiv.org/abs/2104.02395)
- A survey of uncertainty in deep neural networks [[arXiv2021]](https://arxiv.org/abs/2107.03342) - [[GitHub]](https://github.com/JakobCode/UncertaintyInNeuralNetworks_Resources)
- A Survey on Evidential Deep Learning For Single-Pass Uncertainty Estimation [[arXiv2021]](https://arxiv.org/abs/2110.03051)


## Theory

**Conference**

- Top-label calibration and multiclass-to-binary reductions [[ICLR2022]](https://openreview.net/forum?id=WqoBaaPHS-)
- Neural Variational Gradient Descent [[AABI2022]](https://openreview.net/forum?id=oG0vTBw58ic)
- Repulsive Deep Ensembles are Bayesian [[NeurIPS2021]](https://arxiv.org/abs/2106.11642) - [[PyTorch]](https://github.com/ratschlab/repulsive_ensembles)
- Bayesian Optimization with High-Dimensional Outputs [[NeurIPS2021]](https://arxiv.org/abs/2106.12997)
- Residual Pathway Priors for Soft Equivariance Constraints [[NeurIPS2021]](https://arxiv.org/abs/2112.01388)
- Dangers of Bayesian Model Averaging under Covariate Shift [[NeurIPS2021]](https://arxiv.org/abs/2106.11905) - [[TensorFlow]](https://github.com/izmailovpavel/bnn_covariate_shift)
- A Mathematical Analysis of Learning Loss for Active Learning in Regression [[CVPR Workshop2021]](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.html)
- Uncertainty in Gradient Boosting via Ensembles [[ICLR2021]](https://arxiv.org/abs/2006.10562) - [[PyTorch]](https://github.com/yandex-research/GBDT-uncertainty)
- Deep Convolutional Networks as shallow Gaussian Processes [[ICLR2019]](https://arxiv.org/abs/1808.05587)
- On the accuracy of influence functions for measuring group effects [[NeurIPS2018]](https://proceedings.neurips.cc/paper/2019/hash/a78482ce76496fcf49085f2190e675b4-Abstract.html)
- To Trust Or Not To Trust A Classifier [[NeurIPS2018]](https://arxiv.org/abs/1805.11783) - [[Python]](https://github.com/google/TrustScore)
- Understanding Measures of Uncertainty for Adversarial Example Detection [[UAI2018]](https://arxiv.org/abs/1803.08533)

**Journal**

- Multivariate Uncertainty in Deep Learning [[TNNLS2021]](https://arxiv.org/abs/1910.14215)
- A General Framework for Uncertainty Estimation in Deep Learning [[RAL2020]](https://arxiv.org/abs/1907.06890)
- Adaptive nonparametric confidence sets [[The Annals of Statistic (2006)]](https://arxiv.org/abs/math/0605473)

**Arxiv**

- Ensembles for Uncertainty Estimation: Benefits of Prior Functions and Bootstrapping [[arXiv2022]](https://arxiv.org/pdf/2206.03633.pdf)
- Bayesian Model Selection, the Marginal Likelihood, and Generalization [[arXiv2022]](https://arxiv.org/abs/2202.11678)
- Testing for Outliers with Conformal p-values  [[arXiv2021]](https://arxiv.org/abs/2104.08279) - [[Python]](https://github.com/msesia/conditional-conformal-pvalues)
- Efficient Gaussian Neural Processes for Regression [[arXiv2021]](https://arxiv.org/abs/2108.09676)
- Dense Uncertainty Estimation [[arXiv2021]](https://arxiv.org/abs/2110.06427) - [[PyTorch]](https://github.com/JingZhang617/UncertaintyEstimation)
- Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning [[arXiv2020]](https://arxiv.org/abs/2012.09816)
- A higher-order swiss army infinitesimal jackknife [[arXiv2019]](https://arxiv.org/abs/1907.12116)
- With malice towards none: Assessing uncertainty via equalized coverage [[arXiv2019]](https://arxiv.org/abs/1908.05428)


## Bayesian-Methods

**Conference**

- Uncertainty Estimation for Multi-view Data: The Power of Seeing the Whole Picture [[NeurIPS2022]](https://arxiv.org/abs/2210.02676)
- Activation-level uncertainty in deep neural networks [[ICLR2021]](https://openreview.net/forum?id=UvBPbpvHRj-)
- On the Effects of Quantisation on Model Uncertainty in Bayesian Neural Networks [[UAI2021]](https://arxiv.org/abs/2102.11062)
- Learnable uncertainty under Laplace approximations [[UAI2021]](https://proceedings.mlr.press/v161/kristiadi21a.html)
- Bayesian Deep Learning and a Probabilistic Perspective of Generalization [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/file/322f62469c5e3c7dc3e58f5a4d1ea399-Paper.pdf)
- How Good is the Bayes Posterior in Deep Neural Networks Really? [[ICML2020]](http://proceedings.mlr.press/v119/wenzel20a.html)
- Bayesian Uncertainty Estimation for Batch Normalized Deep Networks [[ICML2020]](http://proceedings.mlr.press/v80/teye18a.html)
- Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors [[ICML2020]](http://proceedings.mlr.press/v119/dusenberry20a/dusenberry20a.pdf) - [[TensorFlow]](https://github.com/google/edward2)
- Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks [[ICML2020]](http://proceedings.mlr.press/v119/kristiadi20a/kristiadi20a.pdf)
- TRADI: Tracking deep neural network weight distributions for uncertainty estimation [[ECCV2020]](https://arxiv.org/abs/1912.11316) - [[PyTorch]](https://github.com/giannifranchi/TRADI_Tracking_DNN_weights)
- A Simple Baseline for Bayesian Uncertainty in Deep Learning [[NeurIPS2019]](https://arxiv.org/abs/1902.02476) - [[PyTorch]](https://github.com/wjmaddox/swa_gaussian)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](https://github.com/ezjong/lightprobnets) - [[PyTorch]](https://github.com/ezjong/lightprobnets)
- A Scalable Laplace Approximation for Neural Networks [[ICLR2018]](https://openreview.net/pdf?id=Skdvd2xAZ) - [[Theano]](https://github.com/BB-UCL/Lasagne)
- Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning [[ICML2018]](http://proceedings.mlr.press/v80/depeweg18a.html)
- Weight Uncertainty in Neural Network [[ICML2015]](https://proceedings.mlr.press/v37/blundell15.html)

**Journal**

- Bayesian modeling of uncertainty in low-level vision [[IJCV1990]](https://link.springer.com/article/10.1007%2FBF00126502)

**Arxiv**

- Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification [[arXiv2020]](https://arxiv.org/abs/2012.02818) - [[PyTorch]](https://github.com/giannifranchi/LP_BNN)
- Bayesian Neural Networks with Soft Evidence  [[arXiv2020]](https://arxiv.org/abs/2010.09570#:~:text=Bayes's%20rule%20deals%20with%20hard,has%20actually%20occurred%20or%20not.) - [[PyTorch]](https://github.com/edwardyu/soft-evidence-bnn)
- On Batch Normalisation for Approximate Bayesian Inference [[arXiv2020]](https://openreview.net/pdf?id=SH2tfpm_0LE)
- Bayesian neural network via stochastic gradient descent [[arXiv2020]](https://arxiv.org/abs/2006.08453)


## Ensemble-Methods

**Conference**

- Normalizing Flow Ensembles for Rich Aleatoric and Epistemic Uncertainty Modeling [[AAAI2023]](https://arxiv.org/abs/2302.01312)
- Packed-Ensembles for Efficient Uncertainty Estimation [[ICLR2023]](https://arxiv.org/abs/2210.09184) - [[PyTorch]](https://github.com/ENSTA-U2IS/torch-uncertainty)
- FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation [[NeurIPS2022]](https://arxiv.org/abs/2206.00050)
- Prune and Tune Ensembles: Low-Cost Ensemble Learning With Sparse Independent Subnetworks [[AAAI2022]](https://arxiv.org/abs/2202.11782)
- Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity [[ICLR2022]](https://arxiv.org/abs/2106.14568) - [[PyTorch]](https://github.com/VITA-Group/FreeTickets)
- Robustness via Cross-Domain Ensembles [[ICCV2021]](https://arxiv.org/abs/2103.10919) - [[PyTorch]](https://github.com/EPFL-VILAB/XDEnsembles)
- Masksembles for Uncertainty Estimation [[CVPR2021]](https://nikitadurasov.github.io/projects/masksembles/) - [[PyTorch/TensorFlow]](https://github.com/nikitadurasov/masksembles)
- Uncertainty Quantification and Deep Ensembles [[NeurIPS2021]](https://openreview.net/forum?id=wg_kD_nyAF)
- Uncertainty in Gradient Boosting via Ensembles [[ICLR2021]](https://arxiv.org/abs/2006.10562) - [[PyTorch]](https://github.com/yandex-research/GBDT-uncertainty)
- Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning [[ICLR2020]](https://arxiv.org/abs/2002.06470) - [[PyTorch]](https://github.com/SamsungLabs/pytorch-ensembles)
- Maximizing Overall Diversity for Improved Uncertainty Estimates in Deep Ensembles [[AAAI2020]](https://ojs.aaai.org/index.php/AAAI/article/view/5849)
- Hyperparameter Ensembles for Robustness and Uncertainty Quantification [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/481fbfa59da2581098e841b7afc122f1-Abstract.html)
- Bayesian Deep Ensembles via the Neural Tangent Kernel [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/0b1ec366924b26fc98fa7b71a9c249cf-Abstract.html)
- BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning [[ICLR2020]](https://arxiv.org/abs/2002.06715) - [[TensorFlow]](https://github.com/google/edward2) - [[PyTorch]](https://github.com/giannifranchi/LP_BNN)
- Uncertainty in Neural Networks: Approximately Bayesian Ensembling [[AISTATS 2020]](https://arxiv.org/abs/1810.05546)
- Accurate Uncertainty Estimation and Decomposition in Ensemble Learning [[NeurIPS2019]](https://papers.nips.cc/paper/2019/hash/1cc8a8ea51cd0adddf5dab504a285915-Abstract.html)
- High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach [[ICML2018]](https://arxiv.org/abs/1802.07167) - [[TensorFlow]](https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals)
- Simple and scalable predictive uncertainty estimation using deep ensembles [[NeurIPS2017]](https://arxiv.org/abs/1612.01474)

**Journal**

- One Versus all for deep Neural Network for uncertaInty (OVNNI) quantification [[IEEE Access2021]](https://arxiv.org/abs/2006.00954)

**Arxiv**

- Deep Ensembles Work, But Are They Necessary? [[arXiv2022]](https://arxiv.org/abs/2202.06985)
- On the Usefulness of Deep Ensemble Diversity for Out-of-Distribution Detection [[arXiv2022]](https://arxiv.org/abs/2207.07517)
- Deep Ensemble as a Gaussian Process Approximate Posterior [[arXiv2022]](https://arxiv.org/abs/2205.00163)
- Sequential Bayesian Neural Subnetwork Ensembles [[arXiv2022]](https://arxiv.org/abs/2206.00794)
- Confident Neural Network Regression with Bootstrapped Deep Ensembles [[arXiv2022]](https://arxiv.org/abs/2202.10903) - [[TensorFlow]](https://github.com/LaurensSluyterman/Bootstrapped_Deep_Ensembles)
- Dense Uncertainty Estimation via an Ensemble-based Conditional Latent Variable Model [[arXiv2021]](https://arxiv.org/abs/2111.11055)
- Deep Ensembles: A Loss Landscape Perspective [[arXiv2019]](https://arxiv.org/abs/1912.02757)
- Diversity with Cooperation: Ensemble Methods for Few-Shot Classification [[arXiv2019]](https://arxiv.org/abs/1903.11341)

## Sampling/Dropout-based-Methods

**Conference**

- Efficient Bayesian Uncertainty Estimation for nnU-Net [[MICCAI2022]](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_51)
- Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate [[AAAI2022]](https://arxiv.org/abs/1910.04858v3)
- Dropout Sampling for Robust Object Detection in Open-Set Conditions [[ICRA2018]](https://arxiv.org/abs/1710.06677)
- Test-time data augmentation for estimation of heteroscedastic aleatoric uncertainty in deep neural networks [[MIDL2018]](https://openreview.net/forum?id=rJZz-knjz)
- Concrete Dropout [[NeurIPS2017]](https://arxiv.org/abs/1705.07832)
- Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning [[ICML2016]](https://arxiv.org/abs/1506.02142)

**Journal**
- A General Framework for Uncertainty Estimation in Deep Learning [[Robotics and Automation Letters2020]](https://arxiv.org/pdf/1907.06890.pdf)

**Arxiv**

- SoftDropConnect (SDC) – Effective and Efficient Quantification of the Network Uncertainty in Deep MR Image Analysis [[arXiv2022]](https://arxiv.org/abs/2201.08418)
- Wasserstein Dropout [[arXiv2021]](https://arxiv.org/abs/2012.12687) - [[PyTorch]](https://github.com/fraunhofer-iais/second-moment-loss)

## Auxiliary-Methods/Learning-loss-distributions

**Conference**

- Post-hoc Uncertainty Learning using a Dirichlet Meta-Model [[AAAI2023]](https://arxiv.org/abs/2212.07359)
- Improving the reliability for confidence estimation [[ECCV2022]](https://arxiv.org/abs/2210.06776)
- Gradient-based Uncertainty for Monocular Depth Estimation [[ECCV2022]](https://arxiv.org/abs/2208.02005) - [[PyTorch]](https://github.com/jhornauer/GrUMoDepth)
- BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks [[ECCV2022]](https://arxiv.org/abs/2207.06873) - [[PyTorch]](https://github.com/ExplainableML/BayesCap)
- Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model [[AAAI2022]](https://ojs.aaai.org/index.php/AAAI/article/view/20773)
- Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation [[NeurIPS2022]](https://openreview.net/pdf?id=epjxT_ARZW5)
- Learning Structured Gaussians to Approximate Deep Ensembles [[CVPR2022]](https://arxiv.org/abs/2203.15485)
- SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](https://arxiv.org/abs/2104.02395) - [[PyTorch]](https://github.com/xuanlongORZ/SLURP_uncertainty_estimate)
- Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](https://arxiv.org/abs/2002.05582)
- Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks in Semantic Segmentation [[ICCV2021]](https://arxiv.org/abs/2108.01634) - [[PyTorch]](https://github.com/valeoai/obsnet)
- A Mathematical Analysis of Learning Loss for Active Learning in Regression [[CVPR Workshop2021]](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.html)
- Real-time uncertainty estimation in computer vision via uncertainty-aware distribution distillation [[WACV2021]](https://arxiv.org/abs/2007.15857)
- Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel [[ICLR2020]](https://arxiv.org/abs/1906.00588) - [[TensorFlow]](https://github.com/cognizant-ai-labs/rio-paper)
- Gradients as a Measure of Uncertainty in Neural Networks [[ICIP2020]](https://arxiv.org/abs/2008.08030)
- Learning Loss for Test-Time Augmentation [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html)
- On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](https://arxiv.org/abs/2005.06209) - [[PyTorch]](https://github.com/mattpoggi/mono-uncertainty)
- Addressing failure prediction by learning model confidence [[NeurIPS2019]](https://papers.NeurIPS.cc/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf) - [[PyTorch]](https://github.com/valeoai/ConfidNet)
- Learning loss for active learning [[CVPR2019]](https://arxiv.org/abs/1905.03677) - [[PyTorch]](https://github.com/Mephisto405/Learning-Loss-for-Active-Learning) (unofficial codes)
- Structured Uncertainty Prediction Networks [[CVPR2018]](https://arxiv.org/abs/1802.07079) - [[TensorFlow]](https://github.com/Era-Dorta/tf_mvg)
- Uncertainty estimates and multi-hypotheses networks for optical flow [[ECCV2018]](https://arxiv.org/abs/1802.07095) - [[TensorFlow]](https://github.com/lmb-freiburg/netdef_models)
- Classification uncertainty of deep neural networks based on gradient information [[IAPR Workshop2018]](https://arxiv.org/abs/1805.08440)
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? [[NeurIPS2017]](https://arxiv.org/abs/1703.04977)
- Estimating the Mean and Variance of the Target Probability Distribution [[(ICNN94)]](https://ieeexplore.ieee.org/document/374138)

**Journal**

- Confidence Estimation via Auxiliary Models [[TPAMI2021]](https://arxiv.org/abs/2012.06508)

**Arxiv**

- Instance-Aware Observer Network for Out-of-Distribution Object Segmentation [[arXiv2022]](https://arxiv.org/abs/2207.08782)
- Learning Uncertainty For Safety-Oriented Semantic Segmentation In Autonomous Driving [[arXiv2022]](https://arxiv.org/abs/2105.13688)
- DEUP: Direct Epistemic Uncertainty Prediction [[arXiv2020]](https://arxiv.org/abs/2102.08501)
- Learning Confidence for Out-of-Distribution Detection in Neural Networks [[arXiv2018]](https://arxiv.org/abs/1802.04865)

## Data-augmentation/Generation-based-methods

**Conference**

- Diverse, Global and Amortised Counterfactual Explanations for Uncertainty Estimates [[AAAI2022]](https://arxiv.org/abs/2112.02646)
- PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures [[CVPR2022]](https://arxiv.org/abs/2112.05135)
- Towards efficient feature sharing in MIMO architectures [[CVPR Workshop2022]](https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Sun_Towards_Efficient_Feature_Sharing_in_MIMO_Architectures_CVPRW_2022_paper.html)
- Robust Semantic Segmentation with Superpixel-Mix [[BMVC2021]](https://arxiv.org/abs/2108.00968) - [[PyTorch]](https://github.com/giannifranchi/deeplabv3-superpixelmix)
- MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks [[ICCV2021]](https://arxiv.org/abs/2103.06132) - [[PyTorch]](https://github.com/alexrame/mixmo-pytorch)
- Training independent subnetworks for robust prediction [[ICLR2021]](https://arxiv.org/abs/2010.06610)
- Uncertainty-aware GAN with Adaptive Loss for Robust MRI Image Enhancement  [[ICCV Workshop2021]](https://arxiv.org/pdf/2110.03343.pdf)
- Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning [[ICML2020]](http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf)
- Uncertainty-Aware Deep Classifiers using Generative Models [[AAAI2020]](https://arxiv.org/abs/2006.04183)
- Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [[ECCV2020]](https://arxiv.org/abs/2003.08440) - [[PyTorch]](https://github.com/YingdaXia/SynthCP)
- Detecting the Unexpected via Image Resynthesis [[ICCV2019]](https://arxiv.org/abs/1904.07595) - [[PyTorch]](https://github.com/cvlab-epfl/detecting-the-unexpected)
- On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks [[NeurIPS2019]](https://arxiv.org/abs/1905.11001)
- Deep Anomaly Detection with Outlier Exposure [[ICLR2019]](https://arxiv.org/pdf/1812.04606.pdf)

**Arxiv**

- ZigZag: Universal Sampling-free Uncertainty Estimation Through Two-Step Inference [[arXiv2022]](https://arxiv.org/abs/2211.11435)
- Regularizing Variational Autoencoder with Diversity and Uncertainty Awareness [[arXiv2021]](https://arxiv.org/abs/2110.12381)
- Quantifying uncertainty with GAN-based priors [[arXiv2019]](https://openreview.net/forum?id=HyeAPeBFwS)


## Dirichlet-networks/Evidential-deep-learning

**Conference**

- An Evidential Neural Network Model for Regression Based on Random Fuzzy Numbers [[BELIEF2022]](https://arxiv.org/abs/2208.00647)
- Natural Posterior Network: Deep Bayesian Uncertainty for Exponential Family Distributions [[ICLR2022]](https://arxiv.org/abs/2105.04471) - [[PyTorch]](https://github.com/borchero/natural-posterior-network)
- TBraTS: Trusted Brain Tumor Segmentation [[MICCAI2022]](https://arxiv.org/abs/2206.09309)
- Improving Evidential Deep Learning via Multi-task Learning [[AAAI2022]](https://arxiv.org/abs/2112.09368)
- Trustworthy multimodal regression with mixture of normal-inverse gamma distributions [[NeurIPS2021]](https://arxiv.org/abs/2111.08456)
- Misclassification Risk and Uncertainty Quantification in Deep Classifiers [[WACV2021]](https://openaccess.thecvf.com/content/WACV2021/html/Sensoy_Misclassification_Risk_and_Uncertainty_Quantification_in_Deep_Classifiers_WACV_2021_paper.html)
- Evaluating robustness of predictive uncertainty estimation: Are Dirichlet-based models reliable? [[ICML2021]](http://proceedings.mlr.press/v139/kopetzki21a/kopetzki21a.pdf)
- Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts  [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/0eac690d7059a8de4b48e90f14510391-Abstract.html) - [[PyTorch]](https://github.com/sharpenb/Posterior-Network)
- Being Bayesian about Categorical Probability [[ICML2020]](http://proceedings.mlr.press/v119/joo20a/joo20a.pdf)
- Ensemble Distribution Distillation [[ICLR2020]](https://arxiv.org/abs/1905.00076)
- Conservative Uncertainty Estimation By Fitting Prior Networks [[ICLR2020]](https://openreview.net/forum?id=BJlahxHYDS)
- Noise Contrastive Priors for Functional Uncertainty [[UAI2020]](https://proceedings.mlr.press/v115/hafner20a.html)
- Deep Evidential Regression [[NeurIPS2020]](https://arxiv.org/abs/1910.02600) - [[TensorFlow]](https://github.com/aamini/evidential-deep-learning)
- Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples [[NeurIPS Workshop2020]](https://arxiv.org/abs/2010.10474)
- Uncertainty on Asynchronous Time Event Prediction [[NeurIPS2019]](https://arxiv.org/abs/1911.05503) - [[TensorFlow]](https://github.com/sharpenb/Uncertainty-Event-Prediction)
- Reverse KL-Divergence Training of Prior Networks: Improved Uncertainty and Adversarial Robustness [[NeurIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/7dd2ae7db7d18ee7c9425e38df1af5e2-Abstract.html)
- Quantifying Classification Uncertainty using Regularized Evidential Neural Networks [[AAAI FSS2019]](https://arxiv.org/abs/1910.06864)
- Evidential Deep Learning to Quantify Classification Uncertainty [[NeurIPS2018]](https://arxiv.org/abs/1806.01768) - [[PyTorch]](https://github.com/dougbrion/pytorch-classification-uncertainty)
- Predictive uncertainty estimation via prior networks [[NeurIPS2018]](https://proceedings.neurips.cc/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html)

**Journal**

- Region-Based Evidential Deep Learning to Quantify Uncertainty and Improve Robustness of Brain Tumor Segmentation [[NCA2022]](http://arxiv.org/abs/2208.06038)
- An evidential classifier based on Dempster-Shafer theory and deep learning [[Neurocomputing2021]](https://www.sciencedirect.com/science/article/pii/S0925231221004525) - [[TensorFlow]](https://github.com/tongzheng1992/E-CNN-classifier)
- Evidential fully convolutional network for semantic segmentation [[AppliedIntelligence2021]](https://link.springer.com/article/10.1007/s10489-021-02327-0) - [[TensorFlow]](https://github.com/tongzheng1992/E-FCN)
- Information Aware max-norm Dirichlet networks for predictive uncertainty estimation [[NeuralNetworks2021]](https://arxiv.org/abs/1910.04819#:~:text=Information%20Aware%20Max%2DNorm%20Dirichlet%20Networks%20for%20Predictive%20Uncertainty%20Estimation,-Theodoros%20Tsiligkaridis&text=Precise%20estimation%20of%20uncertainty%20in,prone%20to%20over%2Dconfident%20predictions)
- A neural network classifier based on Dempster-Shafer theory [[IEEETransSMC2000]](https://ieeexplore.ieee.org/abstract/document/833094/)

**Arxiv**

- Uncertainty Estimation by Fisher Information-based Evidential Deep Learning [[arXiv2023]](https://arxiv.org/abs/2303.02045)
- The Unreasonable Effectiveness of Deep Evidential Regression [[arXiv2022]](https://arxiv.org/abs/2205.10060)
- Effective Uncertainty Estimation with Evidential Models for Open-World Recognition [[arXiv2022]](https://openreview.net/pdf?id=NrB52z3eOTY)
- Multivariate Deep Evidential Regression [[arXiv2022]](https://arxiv.org/abs/2104.06135)
- A Survey on Evidential Deep Learning For Single-Pass Uncertainty Estimation [[arXiv2021]](https://arxiv.org/abs/2110.03051)
- Regression Prior Networks [[arXiv2020]](https://arxiv.org/abs/2006.11590)
- Uncertainty estimation in deep learning with application to spoken language assessment[[PhDThesis2019]](https://www.repository.cam.ac.uk/handle/1810/298857)
- Inhibited softmax for uncertainty estimation in neural networks [[arXiv2018]](https://arxiv.org/abs/1810.01861).
- Quantifying Intrinsic Uncertainty in Classification via Deep Dirichlet Mixture Networks [[arXiv2018]](https://arxiv.org/abs/1906.04450)

## Deterministic-Uncertainty-Methods

**Conference**

- Training, Architecture, and Prior for Deterministic Uncertainty Methods [[ICLR Workshop2023]](https://arxiv.org/abs/2303.05796) - [[PyTorch]](https://github.com/orientino/dum-components)
- Latent Discriminant deterministic Uncertainty [[ECCV2022]](https://arxiv.org/abs/2207.10130) - [[PyTorch]](https://github.com/ENSTA-U2IS/LDU)
- Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression [[CoRR2021]](https://arxiv.org/abs/2102.11409)
- Training normalizing flows with the information bottleneck for competitive generative classification [[NeurIPS2020]](https://arxiv.org/abs/2001.06448)
- Simple and principled uncertainty estimation with deterministic deep learning via distance awareness [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html)
- Uncertainty Estimation Using a Single Deep Deterministic Neural Network [[ICML2020]](https://arxiv.org/abs/2003.02037) - [[PyTorch]](https://github.com/y0ast/deterministic-uncertainty-quantification)
- Single-Model Uncertainties for Deep Learning [[NeurIPS2019]](https://arxiv.org/abs/1811.00908) - [[PyTorch]](https://github.com/facebookresearch/SingleModelUncertainty/)
- Sampling-Free Epistemic Uncertainty Estimation Using Approximated Variance Propagation [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/html/Postels_Sampling-Free_Epistemic_Uncertainty_Estimation_Using_Approximated_Variance_Propagation_ICCV_2019_paper.html) - [[PyTorch]](https://github.com/janisgp/Sampling-free-Epistemic-Uncertainty)

**Journal**

- Density estimation in representation space [[EDSMLS2020]](https://arxiv.org/abs/1908.07235)

**Arxiv**

- Deep Deterministic Uncertainty: A Simple Baseline [[arXiv2021]](https://arxiv.org/abs/2102.11582) - [[PyTorch]](https://github.com/omegafragger/DDU)
- Deep Deterministic Uncertainty for Semantic Segmentation [[arXiv2021]](https://arxiv.org/abs/2111.00079)
- On the Practicality of Deterministic Epistemic Uncertainty [[arXiv2021]](https://arxiv.org/abs/2107.00649)
- The Hidden Uncertainty in a Neural Network’s Activations [[arXiv2020]](https://arxiv.org/abs/2012.03082)
- A simple framework for uncertainty in contrastive learning [[arXiv2020]](https://arxiv.org/abs/2010.02038)
- Distance-based Confidence Score for Neural Network Classifiers [[arXiv2017]](https://arxiv.org/abs/1709.09844)

## Quantile-Regression/Predicted-Intervals

**Conference**

- Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging [[ICML2022]](https://arxiv.org/abs/2202.05265) - [[PyTorch]](https://github.com/aangelopoulos/im2im-uq)
- Prediction Intervals: Split Normal Mixture from Quality-Driven Deep Ensembles [[UAI2020]](http://proceedings.mlr.press/v124/saleh-salem20a.html) - [[PyTorch]](https://github.com/tarik/pi-snm-qde)
- Classification with Valid and Adaptive Coverage [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html)
- Conformal Prediction Under Covariate Shift [[NeurIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)
- Conformalized Quantile Regression [[NeurIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
- Single-Model Uncertainties for Deep Learning [[NeurIPS2019]](https://arxiv.org/abs/1811.00908) - [[PyTorch]](https://github.com/facebookresearch/SingleModelUncertainty/)
- High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach [[ICML2018]](https://arxiv.org/abs/1802.07167) - [[TensorFlow]](https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals)

**Journal**

- Scalable Uncertainty Quantification for Deep Operator Networks using Randomized Priors [[CMAME2022]](https://arxiv.org/abs/2203.03048)
- Exploring uncertainty in regression neural networks for construction of prediction intervals [[Neurocomputing2022]](https://www.sciencedirect.com/science/article/abs/pii/S0925231222001102)

**Arxiv**

- Testing for Outliers with Conformal p-values  [[arXiv2021]](https://arxiv.org/abs/2104.08279) - [[Python]](https://github.com/msesia/conditional-conformal-pvalues)
- Interval Neural Networks: Uncertainty Scores [[arXiv2020]](https://arxiv.org/abs/2003.11566)
- Tight Prediction Intervals Using Expanded Interval Minimization [[arXiv2018]](https://arxiv.org/abs/1806.11222)



## Calibration

**Conference**

- Rethinking Confidence Calibration for Failure Prediction [[ECCV2022]](https://link.springer.com/chapter/10.1007/978-3-031-19806-9_30) - [[PyTorch]](https://github.com/Impression2805/FMFP)
- The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration [[CVPR2022]](https://arxiv.org/abs/2111.15430) - [[PyTorch]](https://github.com/by-liu/mbls)
- Calibrating Deep Neural Networks by Pairwise Constraints [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Calibrating_Deep_Neural_Networks_by_Pairwise_Constraints_CVPR_2022_paper.html)
- Top-label calibration and multiclass-to-binary reductions [[ICLR2022]](https://openreview.net/forum?id=WqoBaaPHS-)
- Meta-Calibration: Learning of Model Calibration Using Differentiable Expected Calibration Error [[ICML Workshop2021]](https://arxiv.org/abs/2106.09613) - [[PyTorch]](https://github.com/ondrejbohdal/meta-calibration)
- Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification [[NeurIPS2021]](https://arxiv.org/abs/2011.09588)
- From label smoothing to label relaxation [[AAAI2021]](https://www.aaai.org/AAAI21Papers/AAAI-2191.LienenJ.pdf)
- Calibrating Deep Neural Networks using Focal Loss [[NeurIPS2020]](https://arxiv.org/abs/2002.09437) - [[PyTorch]](https://github.com/torrvision/focal_calibration)
- Stationary activations for uncertainty calibration in deep learning [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/18a411989b47ed75a60ac69d9da05aa5-Abstract.html)
- Confidence-Aware Learning for Deep Neural Networks [[ICML2020]](https://arxiv.org/abs/2007.01458) - [[PyTorch]](https://github.com/daintlab/confidence-aware-learning)
- Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning [[ICML2020]](http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf)
- Regularization via structural label smoothing [[ICML2020]](https://proceedings.mlr.press/v108/li20e.html)
- Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning [[MIDL2020]](http://proceedings.mlr.press/v121/laves20a.html) - [[PyTorch]](https://github.com/mlaves/well-calibrated-regression-uncertainty)
- Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision [[CVPR Workshop2020]](https://arxiv.org/abs/1906.01620) - [[PyTorch]](https://github.com/fregu856/evaluating_bdl)
- When does label smoothing help? [[NeurIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)
- Verified Uncertainty Calibration [[NeurIPS2019]](https://papers.NeurIPS.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html)
- Generalized zero-shot learning with deep calibration network [[NeurIPS2018]](https://proceedings.neurips.cc/paper/2018/hash/1587965fb4d4b5afe8428a4a024feb0d-Abstract.html)
- Measuring Calibration in Deep Learning [[CVPR Workshop2019]](https://arxiv.org/abs/1904.01685)
- Accurate Uncertainties for Deep Learning Using Calibrated Regression [[ICML2018]](https://arxiv.org/abs/1807.00263)
- On calibration of modern neural networks [[ICML2017]](https://arxiv.org/abs/1706.04599)
- On Fairness and Calibration [[NeurIPS2017]](https://arxiv.org/abs/1709.02012)

**Journal**

- Evaluating and Calibrating Uncertainty Prediction in Regression Tasks [[Sensors2022]](https://arxiv.org/abs/1905.11659)
- Calibrated Prediction Intervals for Neural Network Regressors [[IEEE Access 2018]](https://arxiv.org/abs/1803.09546) - [[Python]](https://github.com/cruvadom/Prediction_Intervals)

**Arxiv**

- Towards Understanding Label Smoothing [[arXiv2020]](https://arxiv.org/abs/2006.11653)
- An Investigation of how Label Smoothing Affects Generalization[[arXiv2020]](https://arxiv.org/abs/2010.12648)


## Applications

### Classification and Semantic-Segmentation

**Conference**

- CRISP - Reliable Uncertainty Estimation for Medical Image Segmentation [[MICCAI2022]](https://arxiv.org/abs/2206.07664)
- TBraTS: Trusted Brain Tumor Segmentation [[MICCAI2022]](https://arxiv.org/abs/2206.09309) - [[PyTorch]](https://github.com/cocofeat/tbrats)
- Anytime Dense Prediction with Confidence Adaptivity [[ICLR2022]](https://openreview.net/forum?id=kNKFOXleuC) - [[PyTorch]](https://github.com/liuzhuang13/anytime)
- Robust Semantic Segmentation with Superpixel-Mix [[BMVC2021]](https://arxiv.org/abs/2108.00968) - [[PyTorch]](https://github.com/giannifranchi/deeplabv3-superpixelmix)
- Classification with Valid and Adaptive Coverage [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html)
- DEAL: Difficulty-aware Active Learning for Semantic Segmentation [[ACCV2020]](https://openaccess.thecvf.com/content/ACCV2020/html/Xie_DEAL_Difficulty-aware_Active_Learning_for_Semantic_Segmentation_ACCV_2020_paper.html)
- Human Uncertainty Makes Classification More Robust [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/html/Peterson_Human_Uncertainty_Makes_Classification_More_Robust_ICCV_2019_paper.html)
- Classification uncertainty of deep neural networks based on gradient information [[IAPR Workshop2018]](https://arxiv.org/abs/1805.08440)
- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/html/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.html)
- Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation [[MICCAI2019]](https://arxiv.org/abs/1806.05034) - [[PyTorch]](https://github.com/yulequan/UA-MT)
- A Probabilistic U-Net for Segmentation of Ambiguous Images [[NeurIPS2018]](https://arxiv.org/abs/1806.05034) - [[PyTorch]](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch)
- Evidential Deep Learning to Quantify Classification Uncertainty [[NeurIPS2018]](https://arxiv.org/abs/1806.01768) - [[PyTorch]](https://github.com/dougbrion/pytorch-classification-uncertainty)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](https://arxiv.org/abs/1805.11327) - [[PyTorch]](https://github.com/ezjong/lightprobnets)
- To Trust Or Not To Trust A Classifier [[NeurIPS2018]](https://proceedings.neurips.cc/paper/2018/hash/7180cffd6a8e829dacfc2a31b3f72ece-Abstract.html)
- Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding [[BMVC2017]](https://arxiv.org/abs/1511.02680)

**Journal**

- Explainable machine learning in image classification models: An uncertainty quantification perspective." [[KnowledgeBased2022]](https://www.sciencedirect.com/science/article/pii/S095070512200168X)
- Region-Based Evidential Deep Learning to Quantify Uncertainty and Improve Robustness of Brain Tumor Segmentation [[NCA2022]](https://arxiv.org/abs/2208.06038)

**Arxiv**

- Deep Deterministic Uncertainty for Semantic Segmentation [[arXiv2021]](https://arxiv.org/abs/2111.00079)
- Evaluating Bayesian Deep Learning Methods for Semantic Segmentation [[arXiv2018]](https://arxiv.org/abs/1811.12709)

### Regression

**Conference**

- Variational Depth Networks: Uncertainty-Aware Monocular Self-supervised Depth Estimation [[ECCV Workshop2022]](https://link.springer.com/chapter/10.1007/978-3-031-25085-9_3)
- Uncertainty Quantification in Depth Estimation via Constrained Ordinal Regression [[ECCV2022]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620229.pdf)
- On Monocular Depth Estimation and Uncertainty Quantification using Classification Approaches for Regression [[ICIP2022]](https://arxiv.org/abs/2202.12369)
- Anytime Dense Prediction with Confidence Adaptivity [[ICLR2022]](https://openreview.net/forum?id=kNKFOXleuC) - [[PyTorch]](https://github.com/liuzhuang13/anytime)
- Learning Structured Gaussians to Approximate Deep Ensembles [[CVPR2022]](https://arxiv.org/abs/2203.15485)
- Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate [[AAAI2022]](https://arxiv.org/abs/1910.04858v3)
- Robustness via Cross-Domain Ensembles [[ICCV2021]](https://arxiv.org/abs/2103.10919) - [[PyTorch]](https://github.com/EPFL-VILAB/XDEnsembles)
- SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](https://arxiv.org/abs/2104.02395) - [[PyTorch]](https://github.com/xuanlongORZ/SLURP_uncertainty_estimate)
- Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](https://arxiv.org/abs/2002.05582)
- Deep Evidential Regression [[NeurIPS2020]](https://arxiv.org/abs/1910.02600) - [[TensorFlow]](https://github.com/aamini/evidential-deep-learning)
- Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel [[ICLR2020]](https://arxiv.org/abs/1906.00588) - [[TensorFlow]](https://github.com/cognizant-ai-labs/rio-paper)
- Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning [[MIDL2020]](http://proceedings.mlr.press/v121/laves20a.html) - [[PyTorch]](https://github.com/mlaves/well-calibrated-regression-uncertainty)
- On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](https://arxiv.org/abs/2005.06209) - [[PyTorch]](https://github.com/mattpoggi/mono-uncertainty)
- Fast Uncertainty Estimation for Deep Learning Based Optical Flow [[IROS2020]](https://authors.library.caltech.edu/104758/)
- Inferring Distributions Over Depth from a Single Image [[IROS2019]](https://arxiv.org/abs/1912.06268) - [[TensorFlow]](https://github.com/gengshan-y/monodepth-uncertainty)
- Multi-Task Learning based on Separable Formulation of Depth Estimation and its Uncertainty [[CVPR Workshop2019]](https://openaccess.thecvf.com/content_CVPRW_2019/html/Uncertainty_and_Robustness_in_Deep_Visual_Learning/Asai_Multi-Task_Learning_based_on_Separable_Formulation_of_Depth_Estimation_and_CVPRW_2019_paper.html)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](https://arxiv.org/abs/1805.11327) - [[PyTorch]](https://github.com/ezjong/lightprobnets)
- Uncertainty estimates and multi-hypotheses networks for optical flow [[ECCV2018]](https://arxiv.org/abs/1802.07095) - [[TensorFlow]](https://github.com/lmb-freiburg/netdef_models)
- Accurate Uncertainties for Deep Learning Using Calibrated Regression [[ICML2018]](https://arxiv.org/abs/1807.00263)
- Structured Uncertainty Prediction Networks [[CVPR2018]](https://arxiv.org/abs/1802.07079) - [[TensorFlow]](https://github.com/Era-Dorta/tf_mvg)

**Journal**

- Evaluating and Calibrating Uncertainty Prediction in Regression Tasks [[Sensors2022]](https://arxiv.org/abs/1905.11659)
- Exploring uncertainty in regression neural networks for construction of prediction intervals [[Neurocomputing2022]](https://www.sciencedirect.com/science/article/abs/pii/S0925231222001102)
- Deep Distribution Regression [[Computational Statistics & Data Analysis2021]](https://arxiv.org/abs/1903.06023)
- Calibrated Prediction Intervals for Neural Network Regressors [[IEEE Access 2018]](https://arxiv.org/abs/1803.09546) - [[Python]](https://github.com/cruvadom/Prediction_Intervals)
- Learning a Confidence Measure for Optical Flow [[TPAMI2013]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6261321&casa_token=fYVGhK2pa40AAAAA:XWJdS8zJ4JRw1brCIGiYpzEqMidXTTYVkcKTYnnhSl4ys5pUoHzHO6xsVeGZII9Ir1LAI_3YyfI&tag=1)

**Arxiv**

- How Reliable is Your Regression Model's Uncertainty Under Real-World Distribution Shifts? [[arXiv2023]](https://arxiv.org/abs/2302.03679) - [[PyTorch]](https://github.com/fregu856/regression_uncertainty)
- UncertaINR: Uncertainty Quantification of End-to-End Implicit Neural Representations for Computed Tomographaphy [[arXiv2022]](https://arxiv.org/abs/2202.10847)
- Efficient Gaussian Neural Processes for Regression [[arXiv2021]](https://arxiv.org/abs/2108.09676)
- Wasserstein Dropout [[arXiv2021]](https://arxiv.org/abs/2012.12687) - [[PyTorch]](https://github.com/fraunhofer-iais/second-moment-loss)

### Anomaly-detection and Out-of-Distribution-Detection

**Conference**

- Augmenting Softmax Information for Selective Classification with Out-of-Distribution Data [[ACCV2022]](https://openaccess.thecvf.com/content/ACCV2022/html/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.html)
- Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model [[AAAI2022]](https://ojs.aaai.org/index.php/AAAI/article/view/20773)
- VOS: Learning What You Don't Know by Virtual Outlier Synthesis [[ICLR2022]](https://arxiv.org/abs/2202.01197) - [[PyTorch]](https://github.com/deeplearning-wisc/vos)
- Anomaly Detection via Reverse Distillation from One-Class Embedding [[CVPR2022]](https://arxiv.org/abs/2201.10703#:~:text=Anomaly%20Detection%20via%20Reverse%20Distillation%20from%20One%2DClass%20Embedding,-Hanqiu%20Deng%2C%20Xingyu&text=Knowledge%20distillation%20(KD)%20achieves%20promising,provides%20essential%20evidence%20for%20AD.)
- Towards Total Recall in Industrial Anomaly Detection [[CVPR22]](https://arxiv.org/abs/2106.08265) - [[PyTorch]](https://github.com/hcw-00/PatchCore_anomaly_detection)
- Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection [[WACV2022]](https://arxiv.org/abs/2110.02855) - [[PyTorch]](https://github.com/marco-rudolph/cs-flow)
- On the Importance of Gradients for Detecting Distributional Shifts in the Wild [[NeurIPS2021]](https://arxiv.org/abs/2110.00218)
- Exploring the Limits of Out-of-Distribution Detection [[NeurIPS2021]](https://arxiv.org/abs/2106.03004)
- NAS-OoD: Neural Architecture Search for Out-of-Distribution Generalization [[ICCV2021]](https://arxiv.org/abs/2109.02038)
- NADS: Neural Architecture Distribution Search for Uncertainty Awareness [[ICML2020]](https://arxiv.org/abs/2006.06646)
- Energy-based Out-of-distribution Detection [[NeurIPS2020]](https://arxiv.org/abs/2010.03759?context=cs)
- PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization [[ICPR2020]](https://arxiv.org/abs/2011.08785) - [[PyTorch]](https://github.com/openvinotoolkit/anomalib)
- Detecting out-of-distribution image without learning from out-of-distribution data. [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Hsu_Generalized_ODIN_Detecting_Out-of-Distribution_Image_Without_Learning_From_Out-of-Distribution_Data_CVPR_2020_paper.html)
- Learning Open Set Network with Discriminative Reciprocal Points [[ECCV2020]](https://arxiv.org/abs/2011.00178)
- Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [[ECCV2020]](https://arxiv.org/abs/2003.08440) - [[PyTorch]](https://github.com/YingdaXia/SynthCP)
- Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples [[NeurIPS Workshop2020]](https://arxiv.org/abs/2010.10474)
- Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection [[ICCV2019]](https://arxiv.org/abs/1904.02639) - [[PyTorch]](https://github.com/donggong1/memae-anomaly-detection)
- Detecting the Unexpected via Image Resynthesis [[ICCV2019]](https://arxiv.org/abs/1904.07595) - [[PyTorch]](https://github.com/cvlab-epfl/detecting-the-unexpected)
- Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks [[ICLR2018]](https://arxiv.org/abs/1706.02690)
- A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks [[ICLR2017]](https://arxiv.org/abs/1610.02136) - [[TensorFlow]](https://github.com/hendrycks/error-detection)

**Journal**

- One Versus all for deep Neural Network for uncertaInty (OVNNI) quantification [[IEEE Access2021]](https://arxiv.org/abs/2006.00954)

**Arxiv**

- Generalized out-of-distribution detection: A survey [[arXiv2021]](https://arxiv.org/abs/2110.11334)
- Do We Really Need to Learn Representations from In-domain Data for Outlier Detection? [[arXiv2021]](https://arxiv.org/abs/2105.09270)
- DATE: Detecting Anomalies in Text via Self-Supervision of Transformers [[arXiv2021]](https://arxiv.org/abs/2104.05591)
- Frequentist uncertainty estimates for deep learning [[arXiv2018]](http://bayesiandeeplearning.org/2018/papers/31.pdf)

### Object detection

**Conference**

- Estimating and Evaluating Regression Predictive Uncertainty in Deep Object Detectors [[ICLR2021]](https://openreview.net/forum?id=YLewtnvKgR7)
- Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.pdf) - [[CUDA]](https://github.com/jwchoi384/Gaussian_YOLOv3) - [[PyTorch]](https://github.com/motokimura/PyTorch_Gaussian_YOLOv3) - [[Keras]](https://github.com/xuannianz/keras-GaussianYOLOv3)

### Domain adaptation

**Conference**

- Uncertainty-guided Source-free
Domain Adaptation [[ECCV2022]](https://arxiv.org/pdf/2208.07591.pdf)

# Datasets and Benchmarks

- SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_SHIFT_A_Synthetic_Driving_Dataset_for_Continuous_Multi-Task_Domain_Adaptation_CVPR_2022_paper.html)
- MUAD: Multiple Uncertainties for Autonomous Driving, a benchmark for multiple uncertainty types and tasks [[BMVC2022]](https://arxiv.org/abs/2203.01437) - [[PyTorch]](https://github.com/ENSTA-U2IS/MUAD-Dataset)
- ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding [[ICCV2021]](https://arxiv.org/abs/2104.13395)
- The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection [[IJCV2021]](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf)
- SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation [[NeurIPS2021]](https://arxiv.org/abs/2104.14812)
- Uncertainty Baselines: Benchmarks for Uncertainty & Robustness in Deep Learning [[arXiv2021]](https://arxiv.org/abs/2106.04015) - [[TensorFlow]](https://github.com/google/uncertainty-baselines)
- Curriculum Model Adaptation with Synthetic and Real Data for Semantic Foggy Scene Understanding [[IJCV2020]](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/)
- Benchmarking the Robustness of Semantic Segmentation Models [[CVPR2020]](https://arxiv.org/abs/1908.05005)
- Fishyscapes: A Benchmark for Safe Semantic Segmentation in Autonomous Driving [[ICCV Workshop2019]](https://openaccess.thecvf.com/content_ICCVW_2019/html/ADW/Blum_Fishyscapes_A_Benchmark_for_Safe_Semantic_Segmentation_in_Autonomous_Driving_ICCVW_2019_paper.html)
- Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming [[NeurIPS Workshop2019]](https://arxiv.org/abs/1907.07484) - [[GitHub]](https://github.com/bethgelab/robust-detection-benchmark)
- Semantic Foggy Scene Understanding with Synthetic Data [[IJCV2018]](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)
- Lost and Found: Detecting Small Road Hazards for Self-Driving Vehicles [[IROS2016]](https://arxiv.org/abs/1609.04653)

# Libraries

- Fortuna [[GitHub - JAX]](https://github.com/awslabs/fortuna)
- Bayesian Torch [[GitHub]](https://github.com/IntelLabs/bayesian-torch)
- A Bayesian Neural Network library for PyTorch [[GitHub]](https://github.com/piEsposito/blitz-bayesian-deep-learning)
- TensorFlow Probability [[Website]](https://www.tensorflow.org/probability)
- Uncertainty Toolbox [[GitHub]](https://uncertainty-toolbox.github.io/)
- Mixture Density Networks (MDN) for distribution and uncertainty estimation [[GitHub]](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation)
- OpenOOD: Benchmarking Generalized OOD Detection [[GitHub]](https://github.com/jingkang50/openood)

# Lectures and tutorials

- Uncertainty and Robustness in Deep Learning Workshop in ICML (2020, 2021) [[SlidesLive]](https://slideslive.com/icml-2020/icml-workshop-on-uncertainty-and-robustness-in-deep-learning-udl)
- Yarin Gal: BAYESIAN DEEP LEARNING 101 [[website]](http://www.cs.ox.ac.uk/people/yarin.gal/website/bdl101/)
- MIT 6.S191: Evidential Deep Learning and Uncertainty (2021) [[Youtube]](https://www.youtube.com/watch?v=toTcf7tZK8c)

# Books

- The "Probabilistic Machine-Learning" book series by Kevin Murphy [[Book]](https://probml.github.io/pml-book/)

# Other resources

Awesome conformal prediction [[GitHub]](https://github.com/valeman/awesome-conformal-prediction)

Uncertainty Quantification in Deep Learning [[GitHub]](https://github.com/ahmedmalaa/deep-learning-uncertainty)
