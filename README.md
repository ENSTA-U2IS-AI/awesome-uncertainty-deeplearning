# awesome-uncertainty-deeplearning

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 

This repo is a collection of AWESOME papers/codes/blogs about Uncertainty and Deep learning, including papers, code, etc. Feel free to star and fork.

if you think we missed a paper, please send us an email at:
 gianni.franchi at ensta-paris.fr with the following subject awesome-uncertainty-deeplearning. (tell us where it is published, and send us a GitHub link and arxiv link if they are available)
 
# Contents
<!-- - [awesome-domain-adaptation](#awesome-domain-adaptation) -->
- [Contents](#contents)
- [Papers](#papers)
  - [Survey](#survey)
  - [Theory](#theory)
  - [Ensemble/Bayesian-Methods](#ensemblebayesian-methods)
  - [Sampling/Dropout-based-Methods](#samplingdropout-based-Methods)
  - [Learning-loss-distributions/Auxiliary-Methods](#learning-loss-distributionsauxiliary-methods)
  - [Data-augmentation/Generation-based-Methods](#data-augmentationgeneration-based-methods)
  - [Calibration](#Calibration)
  - [Prior-networks/Evidential-deep-learning](#Prior-networksevidential-deep-learning)
  - [Deterministic-Uncertainty-Methods](#Deterministic-Uncertainty-Methods)
  - [Quantile-Regression/Predicted-Intervals](#quantile-regressionpredicted-intervals)
  - [Applications](#Applications)
    - [Classification and Semantic-Segmentation](#Classification-and-Semantic-Segmentation)
    - [Regression](#Regression)
    - [Anomaly-detection and Out of Distribution Dectection](#Anomaly-detection-and-Out-of-Distribution-Dectection)
- [Datasets and Benchmarks](#Datasets-and-Benchmarks)
- [Library](#Library)
- [Lectures and Tutorials](#Lectures-and-tutorials)
- [Other Resources](#Other-resources)


# Papers
## Survey
**Arxiv**
- A Survey on Uncertainty Reasoning and Quantification for Decision Making: Belief Theory Meets Deep Learning. [[arxiv2022]](https://arxiv.org/abs/2206.05675)
- Ensemble deep learning: A review. [[arxiv2021]](https://arxiv.org/abs/2104.02395)
- A survey of uncertainty in deep neural networks.[[arxiv2021]](https://arxiv.org/abs/2107.03342)[[github]](https://github.com/JakobCode/UncertaintyInNeuralNetworks_Resources)
- A Survey on Evidential Deep Learning For Single-Pass Uncertainty Estimation [[arxiv2021]](https://arxiv.org/abs/2110.03051)


**Conference**
- A Comparison of Uncertainty Estimation Approaches in Deep Learning Components for Autonomous Vehicle Applications[[AISafety2020 Workshop]](https://arxiv.org/abs/2006.15172)

**Journal**
- Predictive inference with the jackknife+." [[The Annals of Statistic(2021)]](https://arxiv.org/abs/1905.02928)
- A review of uncertainty quantification in deep learning: Techniques, applications and challenges [[Information Fusion 2021]](https://www.sciencedirect.com/science/article/pii/S1566253521001081)
- A Survey on Uncertainty Estimation in Deep Learning Classification Systems from a Bayesian Perspective [[ACM2021]](https://dl.acm.org/doi/pdf/10.1145/3477140?casa_token=6fozCYTovlIAAAAA:t5vcjuXCMem1b8iFwaMG4o_YJHTe0wArLtoy9KCbL8Cow0aGEoxSiJans2Kzpm2FSKOg-4ZCDkBa)
- Uncertainty in big data analytics: survey, opportunities, and challenges  [[Journal of Big Data2019]](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0206-3?cv=1)



## Theory
**Arxiv**
- Ensembles for Uncertainty Estimation: Benefits of Prior Functions and Bootstrapping [[arxiv2022]](https://arxiv.org/pdf/2206.03633.pdf)
- Bayesian Model Selection, the Marginal Likelihood, and Generalization [[arxiv2022]](https://arxiv.org/abs/2202.11678)
- Testing for Outliers with Conformal p-values  [[arxiv2021]](https://arxiv.org/abs/2104.08279) [[python]](https://github.com/msesia/conditional-conformal-pvalues)
- Efficient Gaussian Neural Processes for Regression [[arxiv2021]](https://arxiv.org/abs/2108.09676)
- DEUP: Direct Epistemic Uncertainty Prediction [[arxiv2020]](https://arxiv.org/abs/2102.08501)
- A higher-order swiss army infinitesimal jackknife [[arxiv2019]](https://arxiv.org/abs/1907.12116)
- With malice towards none: Assessing uncertainty via equalized coverage [[arxiv2019]](https://arxiv.org/abs/1908.05428)


**Conference**
- Top-label calibration and multiclass-to-binary reductions [[ICLR2022]](https://openreview.net/forum?id=WqoBaaPHS-)
- Neural Variational Gradient Descent [[AABI2022]](https://openreview.net/forum?id=oG0vTBw58ic)
- Bayesian Optimization with High-Dimensional Outputs [[NIPS2021]](https://arxiv.org/abs/2106.12997)
- Residual Pathway Priors for Soft Equivariance Constraints [[NIPS2021]](https://arxiv.org/abs/2112.01388)
- Dangers of Bayesian Model Averaging under Covariate Shift [[NIPS2021]](https://arxiv.org/abs/2106.11905) [[Tensorflow]](https://github.com/izmailovpavel/bnn_covariate_shift)
- A Mathematical Analysis of Learning Loss for Active Learning in Regression [[CVPR2021Workshop]](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.html)
- Uncertainty in Gradient Boosting via Ensembles [[ICLR2021]](https://arxiv.org/abs/2006.10562) [[Pytorch]](https://github.com/yandex-research/GBDT-uncertainty)
- Evidential Deep Learning to Quantify Classification Uncertainty [[NIPS2018]](https://arxiv.org/abs/1806.01768) [[Pytorch]](https://github.com/dougbrion/pytorch-classification-uncertainty)
- On the accuracy of influence functions for measuring group effects [[NIPS2018]](https://proceedings.neurips.cc/paper/2019/hash/a78482ce76496fcf49085f2190e675b4-Abstract.html) 
- To Trust Or Not To Trust A Classifier [[NIPS2018]](https://arxiv.org/abs/1805.11783) [[python]](https://github.com/google/TrustScore)

**Journal**
- Multivariate Uncertainty in Deep Learning [[TNNLS2021]](https://arxiv.org/abs/1910.14215)
- A General Framework for Uncertainty Estimation in Deep Learning [[RAL2020]](https://arxiv.org/abs/1907.06890)
- Adaptive nonparametric confidence sets [[The Annals of Statistic(2006)]](https://arxiv.org/abs/math/0605473)

## Ensemble/Bayesian-Methods
**Arxiv**
- Deep Ensembles Work, But Are They Necessary? [[arxiv2022]](https://arxiv.org/abs/2202.06985)
- On the Usefulness of Deep Ensemble Diversity for Out-of-Distribution Detection [[arxiv2022]](https://arxiv.org/abs/2207.07517)
- Deep Ensemble as a Gaussian Process Approximate Posterior [[arxiv2022]](https://arxiv.org/abs/2205.00163)
- Sequential Bayesian Neural Subnetwork Ensembles [[arxiv2022]](https://arxiv.org/abs/2206.00794)
- FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation [[arxiv]](https://arxiv.org/abs/2206.00050)
- Confident Neural Network Regression with Bootstrapped Deep Ensembles [[arxiv2022]](https://arxiv.org/abs/2202.10903) [[Tensorflow]](https://github.com/LaurensSluyterman/Bootstrapped_Deep_Ensembles)
- Dense Uncertainty Estimation [[arxiv2021]](https://arxiv.org/abs/2110.06427) [[Pytorch]](https://github.com/JingZhang617/UncertaintyEstimation)
- Dense Uncertainty Estimation via an Ensemble-based Conditional Latent Variable Model [[arxiv2021]](https://arxiv.org/abs/2111.11055)
- Repulsive Deep Ensembles are Bayesian [[arxiv2021]](https://arxiv.org/abs/2106.11642)
- Bayesian Neural Networks with Soft Evidence  [[arxiv2020]](https://arxiv.org/abs/2010.09570#:~:text=Bayes's%20rule%20deals%20with%20hard,has%20actually%20occurred%20or%20not.) [[Pytorch]](https://github.com/edwardyu/soft-evidence-bnn)
- On Batch Normalisation for Approximate Bayesian Inference [[arxiv2020]](https://openreview.net/pdf?id=SH2tfpm_0LE)
- Bayesian neural network via stochastic gradient descent [[arxiv2020]](https://arxiv.org/abs/2006.08453)
- Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification [[arxiv2020]](https://arxiv.org/abs/2012.02818) [[Pytorch]](https://github.com/giannifranchi/LP_BNN)
- Deep Ensembles: A Loss Landscape Perspective [[arxiv2019]](https://arxiv.org/abs/1912.02757)
- Diversity with Cooperation: Ensemble Methods for Few-Shot Classification [[arxiv2019]](https://arxiv.org/abs/1903.11341)

**Conference**
- Prune and Tune Ensembles: Low-Cost Ensemble Learning With Sparse Independent Subnetworks [[AAAI2022]](https://arxiv.org/abs/2202.11782)
- Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity [[ICLR2022]](https://arxiv.org/abs/2106.14568) [[Pytorch]](https://github.com/VITA-Group/FreeTickets)
- Activation-level uncertainty in deep neural networks [[ICLR2021]](https://openreview.net/forum?id=UvBPbpvHRj-)
- Robustness via Cross-Domain Ensembles [[ICCV2021]](https://arxiv.org/abs/2103.10919) [[Pytorch]](https://github.com/EPFL-VILAB/XDEnsembles)
- Masksembles for Uncertainty Estimation [[CVPR2021]](https://nikitadurasov.github.io/projects/masksembles/) [[Pytorch/Tensorflow]](https://github.com/nikitadurasov/masksembles)
- On the Effects of Quantisation on Model Uncertainty in Bayesian Neural Networks [[UAI2021]](https://arxiv.org/abs/2102.11062)
- Learnable uncertainty under Laplace approximations [[UAI2021]](https://proceedings.mlr.press/v161/kristiadi21a.html)
- Uncertainty Quantification and Deep Ensembles [[NIPS2021]](https://openreview.net/forum?id=wg_kD_nyAF)
- Real-time uncertainty estimation in computer vision via uncertainty-aware distribution distillation [[WACV2021]](https://arxiv.org/abs/2007.15857)
- Uncertainty in Gradient Boosting via Ensembles [[ICLR2021]](https://arxiv.org/abs/2006.10562) [[Pytorch]](https://github.com/yandex-research/GBDT-uncertainty)
- Ensemble Distribution Distillation [[ICLR2020]](https://arxiv.org/abs/1905.00076)
- Maximizing Overall Diversity for Improved Uncertainty Estimates in Deep Ensembles [[AAAI2020]](https://ojs.aaai.org/index.php/AAAI/article/view/5849)
- Hyperparameter Ensembles for Robustness and Uncertainty Quantification [[NIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/481fbfa59da2581098e841b7afc122f1-Abstract.html)
- Bayesian Uncertainty Estimation for Batch Normalized Deep Networks [[ICML2020]](http://proceedings.mlr.press/v80/teye18a.html)
- BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning [[ICLR2020]](https://arxiv.org/abs/2002.06715) [[Tensorflow]](https://github.com/google/edward2) [[Pytorch]](https://github.com/giannifranchi/LP_BNN)
- A General Framework for Uncertainty Estimation in Deep Learning [[ICRA2020]](https://arxiv.org/pdf/1907.06890.pdf)
- TRADI: Tracking deep neural network weight distributions for uncertainty estimation [[ECCV2020]](https://arxiv.org/abs/1912.11316) [[Pytorch]](https://github.com/giannifranchi/TRADI_Tracking_DNN_weights)
- A Simple Baseline for Bayesian Uncertainty in Deep Learning [[NIPS2019]](https://arxiv.org/abs/1902.02476) [[Pytorch]](https://github.com/wjmaddox/swa_gaussian)
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
- SoftDropConnect (SDC) – Effective and Efficient Quantification of the Network Uncertainty in Deep MR Image Analysis [[arxiv2022]](https://arxiv.org/abs/2201.08418)
- Wasserstein Dropout [[arxiv2021]](https://arxiv.org/abs/2012.12687) [[Pytorch]](https://github.com/fraunhofer-iais/second-moment-loss)

**Conference**
- Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate [[AAAI2022]](https://arxiv.org/abs/1910.04858v3)
- Dropout Sampling for Robust Object Detection in Open-Set Conditions [[ICRA2018]](https://arxiv.org/abs/1710.06677)
- Concrete Dropout [[NIPS2017]](https://arxiv.org/abs/1705.07832)
- Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning [[ICML2016]](https://arxiv.org/abs/1506.02142)

**Journal**
- article

## Learning-loss-distributions/Auxiliary-Methods
**Arxiv**
- Instance-Aware Observer Network for Out-of-Distribution Object Segmentation [[arxiv2022]](https://arxiv.org/abs/2207.08782)
- Learning Uncertainty For Safety-Oriented Semantic Segmentation In Autonomous Driving [[arxiv2022]](https://arxiv.org/abs/2105.13688)
- DEUP: Direct Epistemic Uncertainty Prediction [[arxiv2020]](https://arxiv.org/abs/2102.08501)
- Learning Confidence for Out-of-Distribution Detection in Neural Networks[[arxiv2018]](https://arxiv.org/abs/1802.04865)

**Conference**
- Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model [[AAAI2022]](https://ojs.aaai.org/index.php/AAAI/article/view/20773)
- Gradient-based Uncertainty for Monocular Depth Estimation [[ECCV2022]](https://arxiv.org/abs/2208.02005) [[Pytorch]](https://github.com/jhornauer/GrUMoDepth)
- Learning Structured Gaussians to Approximate Deep Ensembles [[CVPR2022]](https://arxiv.org/abs/2203.15485)
- SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](https://arxiv.org/abs/2104.02395) [[Pytorch]](https://github.com/xuanlongORZ/SLURP_uncertainty_estimate) 
- Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](https://arxiv.org/abs/2002.05582)
- A Mathematical Analysis of Learning Loss for Active Learning in Regression [[CVPR2021Workshop]](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.html)
- Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning [[ICLR202]](https://arxiv.org/abs/2002.06470) [[Pytorch]](https://github.com/SamsungLabs/pytorch-ensembles)
- Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel [[ICLR2020]](https://arxiv.org/abs/1906.00588) [[Tensorflow]](https://github.com/cognizant-ai-labs/rio-paper)
- Gradients as a Measure of Uncertainty in Neural Networks [[ICIP2020]](https://arxiv.org/abs/2008.08030)
- Learning Loss for Test-Time Augmentation [[NIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html)
- On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](https://arxiv.org/abs/2005.06209) [[Pytorch]](https://github.com/mattpoggi/mono-uncertainty)
- Addressing failure prediction by learning model confidence [[NeurIPS2019]](https://papers.nips.cc/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf)[[Pytorch]](https://github.com/valeoai/ConfidNet)
- Learning loss for active learning [[CVPR2019]](https://arxiv.org/abs/1905.03677) [[Pytorch]](https://github.com/Mephisto405/Learning-Loss-for-Active-Learning) (unofficial codes)
- Structured Uncertainty Prediction Networks [[CVPR2018]](https://arxiv.org/abs/1802.07079) [[Tensorflow]](https://github.com/Era-Dorta/tf_mvg)
- Classification uncertainty of deep neural networks based on gradient information [[IAPR Workshop2018]](https://arxiv.org/abs/1805.08440)
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?[[NIPS2017]](https://arxiv.org/abs/1703.04977) 
- Estimating the Mean and Variance of the Target Probability Distribution [[(ICNN94)]](https://ieeexplore.ieee.org/document/374138)

**Journal**
- Confidence Estimation via Auxiliary Models [[TPAMI2021]](https://arxiv.org/abs/2012.06508)



## Data-augmentation/Generation-based-methods
**Arxiv**
- Diverse, Global and Amortised Counterfactual Explanations for Uncertainty Estimates [[arxiv2021]](https://arxiv.org/abs/2112.02646)
- Regularizing Variational Autoencoder with Diversity and Uncertainty Awareness [[arxiv2021]](https://arxiv.org/abs/2110.12381)
- PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures." [[arxiv2021]](https://arxiv.org/abs/2112.05135)
- Quantifying uncertainty with GAN-based priors [[arxiv2019]](https://openreview.net/forum?id=HyeAPeBFwS)


**Conference**
- Towards efficient feature sharing in MIMO architectures [[CVPRW2022]](https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Sun_Towards_Efficient_Feature_Sharing_in_MIMO_Architectures_CVPRW_2022_paper.html)
- Robust Semantic Segmentation with Superpixel-Mix [[BMVC2021]](https://arxiv.org/abs/2108.00968) [[Pytorch]](https://github.com/giannifranchi/deeplabv3-superpixelmix)
- MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks [[ICCV2021]](https://arxiv.org/abs/2103.06132) [[Pytorch]](https://github.com/alexrame/mixmo-pytorch)
- Training independent subnetworks for robust prediction [[ICLR2021]](https://arxiv.org/abs/2010.06610)
- Uncertainty-aware GAN with Adaptive Loss for Robust MRI Image Enhancement  [[ICCVWorkshop2021]](https://arxiv.org/pdf/2110.03343.pdf)
- Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning [[ICML2020]](http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf)
- Uncertainty-Aware Deep Classifiers using Generative Models [[AAAI2020]](https://arxiv.org/abs/2006.04183)
- Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [[ECCV2020]](https://arxiv.org/abs/2003.08440) [[Pytorch]](https://github.com/YingdaXia/SynthCP)
- Detecting the Unexpected via Image Resynthesis [[ICCV2019]](https://arxiv.org/abs/1904.07595) [[Pytorch]](https://github.com/cvlab-epfl/detecting-the-unexpected)
- On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks [[NIPS2019]](https://arxiv.org/abs/1905.11001)

**Journal**
- article


## Calibration
**Arxiv**
- Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification [[arxiv2021]](https://arxiv.org/abs/2011.09588)
- The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration [[arxiv2021]](https://arxiv.org/abs/2111.15430)[[Pytorch]](https://github.com/by-liu/mbls)
- Evaluating and Calibrating Uncertainty Prediction in Regression Tasks [[arxiv2020]](https://arxiv.org/abs/1905.11659)
- Towards Understanding Label Smoothing [[arxiv2020]](https://arxiv.org/abs/2006.11653)
- An Investigation of how Label Smoothing Affects Generalization[[arxiv2020]](https://arxiv.org/abs/2010.12648)
- On Fairness and Calibration[[arxiv2017]](https://arxiv.org/abs/1709.02012)



**Conference**
- Calibrating Deep Neural Networks by Pairwise Constraints [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Calibrating_Deep_Neural_Networks_by_Pairwise_Constraints_CVPR_2022_paper.html)
- Top-label calibration and multiclass-to-binary reductions [[ICLR2022]](https://openreview.net/forum?id=WqoBaaPHS-)
- From label smoothing to label relaxation [[AAAI2021]](https://www.aaai.org/AAAI21Papers/AAAI-2191.LienenJ.pdf)
- Calibrating Deep Neural Networks using Focal Loss [[NIPS2020]](https://arxiv.org/abs/2002.09437) [[Pytorch]](https://github.com/torrvision/focal_calibration)
- Stationary activations for uncertainty calibration in deep learning [[NIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/18a411989b47ed75a60ac69d9da05aa5-Abstract.html)
- Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning [[ICML2020]](http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf)
- Regularization via structural label smoothing [[ICML2020]](https://proceedings.mlr.press/v108/li20e.html)
- Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning [[MIDL2020]](http://proceedings.mlr.press/v121/laves20a.html) [[Pytorch]](https://github.com/mlaves/well-calibrated-regression-uncertainty)
- Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision [[CVPRW2020]](https://arxiv.org/abs/1906.01620) [[Pytorch]](https://github.com/fregu856/evaluating_bdl)
- When does label smoothing help? [[NIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)
- Verified Uncertainty Calibration [[NIPS2019]](https://papers.nips.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html)
- Generalized zero-shot learning with deep calibration network [[NIPS2018]](https://proceedings.neurips.cc/paper/2018/hash/1587965fb4d4b5afe8428a4a024feb0d-Abstract.html)
- Measuring Calibration in Deep Learning [[CVPRW2019]](https://arxiv.org/abs/1904.01685)
- Accurate Uncertainties for Deep Learning Using Calibrated Regression [[ICML2018]](https://arxiv.org/abs/1807.00263)
- On calibration of modern neural networks. [[ICML2017]](https://arxiv.org/abs/1706.04599)

**Journal**
- Calibrated Prediction Intervals for Neural Network Regressors [[IEEE Access 2018]](https://arxiv.org/abs/1803.09546)[[Python]](https://github.com/cruvadom/Prediction_Intervals)


## Prior-networks/Evidential-deep-learning
**Arxiv**
- Region-Based Evidential Deep Learning to Quantify Uncertainty and Improve Robustness of Brain Tumor Segmentation [[arxiv2022]](http://arxiv.org/abs/2208.06038)
- Effective Uncertainty Estimation with Evidential Models for Open-World Recognition [[arxiv2022]](https://openreview.net/forum?id=NrB52z3eOTY)
- The Unreasonable Effectiveness of Deep Evidential Regression [[arxiv2022]](https://arxiv.org/abs/2205.10060)
- Effective Uncertainty Estimation with Evidential Models for Open-World Recognition [[arxiv2022]](https://openreview.net/pdf?id=NrB52z3eOTY)
- Multivariate Deep Evidential Regression [[arxiv2022]](https://arxiv.org/abs/2104.06135)
- A Survey on Evidential Deep Learning For Single-Pass Uncertainty Estimation [[arxiv2021]](https://arxiv.org/abs/2110.03051)
- Regression Prior Networks [[arxiv2020]](https://arxiv.org/abs/2006.11590)
- Uncertainty estimation in deep learning with application to spoken language assessment[[phdthesis2019]](https://www.repository.cam.ac.uk/handle/1810/298857)
- Inhibited softmax for uncertainty estimation in neural networks [[arxiv2018]](https://arxiv.org/abs/1810.01861).
- Quantifying Intrinsic Uncertainty in Classification via Deep Dirichlet Mixture Networks [[arxiv2018]](https://arxiv.org/abs/1906.04450)

**Conference**
- Natural Posterior Network: Deep Bayesian Uncertainty for Exponential Family Distributions [[ICLR2022]](https://arxiv.org/abs/2105.04471) [[Pytorch]](https://github.com/borchero/natural-posterior-network)
- TBraTS: Trusted Brain Tumor Segmentation [[MICCAI2022]](https://arxiv.org/abs/2206.09309)
- Improving Evidential Deep Learning via Multi-task Learning [[AAAI2022]](https://arxiv.org/abs/2112.09368)
- Misclassification Risk and Uncertainty Quantification in Deep Classifiers [[WACV2021]](https://openaccess.thecvf.com/content/WACV2021/html/Sensoy_Misclassification_Risk_and_Uncertainty_Quantification_in_Deep_Classifiers_WACV_2021_paper.html)
- Evaluating robustness of predictive uncertainty estimation: Are Dirichlet-based models reliable? [[ICML2021]](http://proceedings.mlr.press/v139/kopetzki21a/kopetzki21a.pdf)
- Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts  [[NIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/0eac690d7059a8de4b48e90f14510391-Abstract.html) [[Pytorch]](https://github.com/sharpenb/Posterior-Network)
- Conservative Uncertainty Estimation By Fitting Prior Networks [[ICLR2020]](https://openreview.net/forum?id=BJlahxHYDS)
- Noise Contrastive Priors for Functional Uncertainty [[UAI2020]](https://proceedings.mlr.press/v115/hafner20a.html)
- Deep Evidential Regression [[NIPS2020]](https://arxiv.org/abs/1910.02600) [[Tensorflow]](https://github.com/aamini/evidential-deep-learning)
- Reverse KL-Divergence Training of Prior Networks: Improved Uncertainty and Adversarial Robustness [[NIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/7dd2ae7db7d18ee7c9425e38df1af5e2-Abstract.html)
- Quantifying Classification Uncertainty using Regularized Evidential Neural Networks [[AAAI FSS2019]](https://arxiv.org/abs/1910.06864)
- Evidential Deep Learning to Quantify Classification Uncertainty [[NIPS2018]](https://arxiv.org/abs/1806.01768) [[Pytorch]](https://github.com/dougbrion/pytorch-classification-uncertainty)
- Predictive uncertainty estimation via prior networks [[NIPS2018]](https://proceedings.neurips.cc/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html)

**Journal**
- An evidential classifier based on Dempster-Shafer theory and deep learning [[Neurocomputing2021]](https://www.sciencedirect.com/science/article/pii/S0925231221004525) [[Tensorflow]](https://github.com/tongzheng1992/E-CNN-classifier)
- Evidential fully convolutional network for semantic segmentation [[AppliedIntelligence2021]](https://link.springer.com/article/10.1007/s10489-021-02327-0) [[Tensorflow]](https://github.com/tongzheng1992/E-FCN)
- Information Aware max-norm Dirichlet networks for predictive uncertainty estimation [[NeuralNetworks2021]](https://arxiv.org/abs/1910.04819#:~:text=Information%20Aware%20Max%2DNorm%20Dirichlet%20Networks%20for%20Predictive%20Uncertainty%20Estimation,-Theodoros%20Tsiligkaridis&text=Precise%20estimation%20of%20uncertainty%20in,prone%20to%20over%2Dconfident%20predictions)
- A neural network classifier based on Dempster-Shafer theory [[IEEETransSMC2000]](https://ieeexplore.ieee.org/abstract/document/833094/)

## Deterministic-Uncertainty-Methods

**Arxiv**
- Deep Deterministic Uncertainty: A Simple Baseline [[arxiv2021]](https://arxiv.org/abs/2102.11582) [[Pytorch]](https://github.com/omegafragger/DDU)
- Deep Deterministic Uncertainty for Semantic Segmentation [[arxiv2021]](https://arxiv.org/abs/2111.00079)
- On the Practicality of Deterministic Epistemic Uncertainty [[arxiv2021]](https://arxiv.org/abs/2107.00649)
- The Hidden Uncertainty in a Neural Network’s Activations [[arxiv2020]](https://arxiv.org/abs/2012.03082)
-  A simple framework for uncertainty in contrastive learning [[arxiv2020]](https://arxiv.org/abs/2010.02038)
- Density estimation in representation space [[arxiv2019]](https://arxiv.org/abs/1908.07235)
- Distance-based Confidence Score for Neural Network Classifiers [[arxiv2017]](https://arxiv.org/abs/1709.09844)

**Conference**
- Latent Discriminant deterministic Uncertainty [[ECCV2022]](https://arxiv.org/abs/2207.10130) [[Pytorch]](https://github.com/ENSTA-U2IS/LDU)
- Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression [[CoRR2021]](https://arxiv.org/abs/2102.11409)
- Training normalizing flows with the information bottleneck for competitive generative classification [[NIPS2020]](https://arxiv.org/abs/2001.06448)
- Simple and principled uncertainty estimation with deterministic deep learning via distance awareness [[NIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html)
- Uncertainty Estimation Using a Single Deep Deterministic Neural Network [[ICML2020]](https://arxiv.org/abs/2003.02037) [[Pytorch]](https://github.com/y0ast/deterministic-uncertainty-quantification)
- Single-Model Uncertainties for Deep Learning [[NIPS2019]](https://arxiv.org/abs/1811.00908) [[Pytorch]](https://github.com/facebookresearch/SingleModelUncertainty/)
- Sampling-Free Epistemic Uncertainty Estimation Using Approximated Variance Propagation [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/html/Postels_Sampling-Free_Epistemic_Uncertainty_Estimation_Using_Approximated_Variance_Propagation_ICCV_2019_paper.html) [[Pytorch]](https://github.com/janisgp/Sampling-free-Epistemic-Uncertainty)

**Journal**
- article

## Quantile-Regression/Predicted-Intervals

**Arxiv**
- Scalable Uncertainty Quantification for Deep Operator Networks using Randomized Priors.[[Arxiv2022]](https://arxiv.org/abs/2203.03048)
- Testing for Outliers with Conformal p-values  [[arxiv2021]](https://arxiv.org/abs/2104.08279) [[python]](https://github.com/msesia/conditional-conformal-pvalues)
- Interval Neural Networks: Uncertainty Scores [[arxiv2020]](https://arxiv.org/abs/2003.11566)
- Tight Prediction Intervals Using Expanded Interval Minimization [[arxiv2018]](https://arxiv.org/abs/1806.11222)


**Conference**
- Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging [[ICML2022]](https://arxiv.org/abs/2202.05265) [[PyTorch]](https://github.com/aangelopoulos/im2im-uq)
- Prediction Intervals: Split Normal Mixture from Quality-Driven Deep Ensembles [[UAI2020]](http://proceedings.mlr.press/v124/saleh-salem20a.html) [[Pytorch]](https://github.com/tarik/pi-snm-qde)
- Classification with Valid and Adaptive Coverage [[NIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html)
- Conformal Prediction Under Covariate Shift [[NIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)
- Conformalized Quantile Regression [[NIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
- Single-Model Uncertainties for Deep Learning [[NIPS2019]](https://arxiv.org/abs/1811.00908) [[Pytorch]](https://github.com/facebookresearch/SingleModelUncertainty/)
- High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach [[ICML2018]](https://arxiv.org/abs/1802.07167) [[Tensorflow]](https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals)

**Journal**
- Exploring uncertainty in regression neural networks for construction of prediction intervals [[Neurocomputing2022]](https://www.sciencedirect.com/science/article/abs/pii/S0925231222001102)

## Applications


### Classification and Semantic-Segmentation
**Arxiv**
- Region-Based Evidential Deep Learning to Quantify Uncertainty and Improve Robustness of Brain Tumor Segmentation [[arxiv2022]](https://arxiv.org/abs/2208.06038)
- Deep Deterministic Uncertainty for Semantic Segmentation [[arxiv2021]](https://arxiv.org/abs/2111.00079)
- Evaluating Bayesian Deep Learning Methods for Semantic Segmentation [[arxiv2018]](https://arxiv.org/abs/1811.12709)


**Conference**
- CRISP - Reliable Uncertainty Estimation for Medical Image Segmentation [[MICCAI2022]](https://arxiv.org/abs/2206.07664)
- TBraTS: Trusted Brain Tumor Segmentation [[MICCAI2022]](https://arxiv.org/abs/2206.09309) [[Pytorch]](https://github.com/cocofeat/tbrats)
- Anytime Dense Prediction with Confidence Adaptivity [[ICLR2022]](https://openreview.net/forum?id=kNKFOXleuC) [[Pytorch]](https://github.com/liuzhuang13/anytime)
- Robust Semantic Segmentation with Superpixel-Mix [[BMVC2021]](https://arxiv.org/abs/2108.00968) [[Pytorch]](https://github.com/giannifranchi/deeplabv3-superpixelmix)
- Classification with Valid and Adaptive Coverage [[NIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html)
- DEAL: Difficulty-aware Active Learning for Semantic Segmentation [[ACCV2020]](https://openaccess.thecvf.com/content/ACCV2020/html/Xie_DEAL_Difficulty-aware_Active_Learning_for_Semantic_Segmentation_ACCV_2020_paper.html)
- Human Uncertainty Makes Classification More Robust [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/html/Peterson_Human_Uncertainty_Makes_Classification_More_Robust_ICCV_2019_paper.html)
- Classification uncertainty of deep neural networks based on gradient information [[IAPR Workshop2018]](https://arxiv.org/abs/1805.08440)
- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/html/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.html)
- Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation [[MICCAI2019]](https://arxiv.org/abs/1806.05034)[[Pytorch]](https://github.com/yulequan/UA-MT)
- A Probabilistic U-Net for Segmentation of Ambiguous Images [[NIPS2018]](https://arxiv.org/abs/1806.05034) [[Pytorch]](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch)
- Evidential Deep Learning to Quantify Classification Uncertainty [[NIPS2018]](https://arxiv.org/abs/1806.01768) [[Pytorch]](https://github.com/dougbrion/pytorch-classification-uncertainty)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](https://arxiv.org/abs/1805.11327)[[Pytorch]](https://github.com/ezjong/lightprobnets)
- To Trust Or Not To Trust A Classifier [[NIPS2018]](https://proceedings.neurips.cc/paper/2018/hash/7180cffd6a8e829dacfc2a31b3f72ece-Abstract.html)
- Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding [[BMVC2017]](https://arxiv.org/abs/1511.02680)

**Journal**
- Explainable machine learning in image classification models: An uncertainty quantification perspective." [[KnowledgeBased2022]](https://www.sciencedirect.com/science/article/pii/S095070512200168X)


### Regression
**Arxiv**
- UncertaINR: Uncertainty Quantification of End-to-End Implicit Neural Representations for Computed TomographarXiv [[arxiv2022]](https://arxiv.org/abs/2202.10847)
- Efficient Gaussian Neural Processes for Regression [[arxiv2021]](https://arxiv.org/abs/2108.09676)
- Wasserstein Dropout [[arxiv2021]](https://arxiv.org/abs/2012.12687) [[Pytorch]](https://github.com/fraunhofer-iais/second-moment-loss)
- Evaluating and Calibrating Uncertainty Prediction in Regression Tasks [[arxiv2020]](https://arxiv.org/abs/1905.11659)

**Conference**
- On Monocular Depth Estimation and Uncertainty Quantification using Classification Approaches for Regression [[ICIP2022]](https://arxiv.org/abs/2202.12369)
- Anytime Dense Prediction with Confidence Adaptivity [[ICLR2022]](https://openreview.net/forum?id=kNKFOXleuC) [[Pytorch]](https://github.com/liuzhuang13/anytime)
- Learning Structured Gaussians to Approximate Deep Ensembles [[CVPR2022]](https://arxiv.org/abs/2203.15485)
- Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate [[AAAI2022]](https://arxiv.org/abs/1910.04858v3)
- Robustness via Cross-Domain Ensembles [[ICCV2021]](https://arxiv.org/abs/2103.10919) [[Pytorch]](https://github.com/EPFL-VILAB/XDEnsembles)
- SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](https://arxiv.org/abs/2104.02395) [[Pytorch]](https://github.com/xuanlongORZ/SLURP_uncertainty_estimate) 
- Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](https://arxiv.org/abs/2002.05582)
- Deep Evidential Regression [[NIPS2020]](https://arxiv.org/abs/1910.02600) [[Tensorflow]](https://github.com/aamini/evidential-deep-learning)
- Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel [[ICLR2020]](https://arxiv.org/abs/1906.00588) [[Tensorflow]](https://github.com/cognizant-ai-labs/rio-paper)
- Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning [[MIDL2020]](http://proceedings.mlr.press/v121/laves20a.html) [[Pytorch]](https://github.com/mlaves/well-calibrated-regression-uncertainty)
- On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](https://arxiv.org/abs/2005.06209) [[Pytorch]](https://github.com/mattpoggi/mono-uncertainty)
- Fast Uncertainty Estimation for Deep Learning Based Optical Flow [[IROS2020]](https://authors.library.caltech.edu/104758/)
- Inferring Distributions Over Depth from a Single Image [[IROS2019]](https://arxiv.org/abs/1912.06268) [[Tensorflow]](https://github.com/gengshan-y/monodepth-uncertainty)
- Multi-Task Learning based on Separable Formulation of Depth Estimation and its Uncertainty [[CVPRW]](https://openaccess.thecvf.com/content_CVPRW_2019/html/Uncertainty_and_Robustness_in_Deep_Visual_Learning/Asai_Multi-Task_Learning_based_on_Separable_Formulation_of_Depth_Estimation_and_CVPRW_2019_paper.html)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](https://arxiv.org/abs/1805.11327)[[Pytorch]](https://github.com/ezjong/lightprobnets)
- Uncertainty estimates and multi-hypotheses networks for optical flow [[ECCV2018]](https://arxiv.org/abs/1802.07095) [[Tensorflow]](https://github.com/lmb-freiburg/netdef_models)
- Accurate Uncertainties for Deep Learning Using Calibrated Regression [[ICML2018]](https://arxiv.org/abs/1807.00263)
- Structured Uncertainty Prediction Networks [[CVPR2018]](https://arxiv.org/abs/1802.07079) [[Tensorflow]](https://github.com/Era-Dorta/tf_mvg)

**Journal**
- Exploring uncertainty in regression neural networks for construction of prediction intervals [[Neurocomputing2022]](https://www.sciencedirect.com/science/article/abs/pii/S0925231222001102)
- Calibrated Prediction Intervals for Neural Network Regressors [[IEEE Access 2018]](https://arxiv.org/abs/1803.09546)[[Python]](https://github.com/cruvadom/Prediction_Intervals)
- Learning a Confidence Measure for Optical Flow [[TPAMI2013]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6261321&casa_token=fYVGhK2pa40AAAAA:XWJdS8zJ4JRw1brCIGiYpzEqMidXTTYVkcKTYnnhSl4ys5pUoHzHO6xsVeGZII9Ir1LAI_3YyfI&tag=1)


### Anomaly-detection and Out-of-Distribution-Dectection
**Arxiv**
- Generalized out-of-distribution detection: A survey [[arxiv2021]](https://arxiv.org/abs/2110.11334)
- Towards Total Recall in Industrial Anomaly Detection [[arxiv2021]](https://arxiv.org/abs/2106.08265) [[Pytorch]](https://github.com/hcw-00/PatchCore_anomaly_detection)
- Do We Really Need to Learn Representations from In-domain Data for Outlier Detection? [[arxiv2021]](https://arxiv.org/abs/2105.09270)
- Exploring the Limits of Out-of-Distribution Detection [[arxiv2021]](https://arxiv.org/abs/2106.03004)
- DATE: Detecting Anomalies in Text via Self-Supervision of Transformers [[arxiv2021]](https://arxiv.org/abs/2104.05591)
- Frequentist uncertainty estimates for deep learning [[arxiv2018]](http://bayesiandeeplearning.org/2018/papers/31.pdf)


**Conference**
- Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model [[AAAI2022]](https://ojs.aaai.org/index.php/AAAI/article/view/20773)
- VOS: Learning What You Don't Know by Virtual Outlier Synthesis [[ICLR2022]](https://arxiv.org/abs/2202.01197) [[Pytorch]](https://github.com/deeplearning-wisc/vos)
- Anomaly Detection via Reverse Distillation from One-Class Embedding [[CVPR2022]](https://arxiv.org/abs/2201.10703#:~:text=Anomaly%20Detection%20via%20Reverse%20Distillation%20from%20One%2DClass%20Embedding,-Hanqiu%20Deng%2C%20Xingyu&text=Knowledge%20distillation%20(KD)%20achieves%20promising,provides%20essential%20evidence%20for%20AD.)
- Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection [[WACV2022]](https://arxiv.org/abs/2110.02855) [[Pytorch]](https://github.com/marco-rudolph/cs-flow)
- On the Importance of Gradients for Detecting Distributional Shifts in the Wild [[NeurIPS2021]](https://arxiv.org/abs/2110.00218)
- Energy-based Out-of-distribution Detection [[NIPS2020]](https://arxiv.org/abs/2010.03759?context=cs)
- PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization [[ICPR2020]](https://arxiv.org/abs/2011.08785) [[Pytorch]](https://github.com/openvinotoolkit/anomalib)
- Detecting out-of-distribution image without learning from out-of-distribution data. [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Hsu_Generalized_ODIN_Detecting_Out-of-Distribution_Image_Without_Learning_From_Out-of-Distribution_Data_CVPR_2020_paper.html)
- Learning Open Set Network with Discriminative Reciprocal Points [[ECCV2020]](https://arxiv.org/abs/2011.00178)
- Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [[ECCV2020]](https://arxiv.org/abs/2003.08440)[[Pytorch]](https://github.com/YingdaXia/SynthCP)
- Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples [[NIPS workshop2020]](https://arxiv.org/abs/2010.10474)
- Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection [[ICCV2019]](https://arxiv.org/abs/1904.02639) [[Pytorch]](https://github.com/donggong1/memae-anomaly-detection) 
- Detecting the Unexpected via Image Resynthesis [[ICCV2019]](https://arxiv.org/abs/1904.07595)[[Pytorch]](https://github.com/cvlab-epfl/detecting-the-unexpected) 
- Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks [[ICLR2018]](https://arxiv.org/abs/1706.02690)
- A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks [[ICLR2017]](https://arxiv.org/abs/1610.02136) [[Tensorflow]](https://github.com/hendrycks/error-detection)

**Journal**
- One Versus all for deep Neural Network for uncertaInty (OVNNI) quantification [[IEEE Access2021]](https://arxiv.org/abs/2006.00954)



# Datasets and Benchmarks
- SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_SHIFT_A_Synthetic_Driving_Dataset_for_Continuous_Multi-Task_Domain_Adaptation_CVPR_2022_paper.html)
- MUAD: Multiple Uncertainties for Autonomous Driving benchmark for multiple uncertainty types and tasks [[arxiv2022]](https://arxiv.org/abs/2203.01437)
- ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding [[ICCV2021]](https://arxiv.org/abs/2104.13395)
- The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection [[IJCV2021]](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf)
- SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation [[NIPS2021]](https://arxiv.org/abs/2104.14812)
- Uncertainty Baselines: Benchmarks for Uncertainty & Robustness in Deep Learning [[arxiv2021]](https://arxiv.org/abs/2106.04015)[[Tensorflow]](https://github.com/google/uncertainty-baselines)
- Curriculum Model Adaptation with Synthetic and Real Data for Semantic Foggy Scene Understanding [[IJCV2020]](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/)
- Fishyscapes: A Benchmark for Safe Semantic Segmentation in Autonomous Driving [[ICCVW2019]](https://openaccess.thecvf.com/content_ICCVW_2019/html/ADW/Blum_Fishyscapes_A_Benchmark_for_Safe_Semantic_Segmentation_in_Autonomous_Driving_ICCVW_2019_paper.html)
- Semantic Foggy Scene Understanding with Synthetic Data [[IJCV2018]](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)
- Lost and Found: Detecting Small Road Hazards for Self-Driving Vehicles [[IROS2016]](https://arxiv.org/abs/1609.04653)
# Library
- Bayesian Torch [[github]](https://github.com/IntelLabs/bayesian-torch)
- A Bayesian Neural Network library for PyTorch [[github]](https://github.com/piEsposito/blitz-bayesian-deep-learning)
- Uncertainty Toolbox [[github]](https://uncertainty-toolbox.github.io/)
- Mixture Density Networks (MDN) for distribution and uncertainty estimation [[github]](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation)

# Lectures-and-tutorials
-  Uncertainty and Robustness in Deep Learning Workshop in ICML (2020, 2021) [[SlidesLive]](https://slideslive.com/icml-2020/icml-workshop-on-uncertainty-and-robustness-in-deep-learning-udl)
- Yarin Gal: BAYESIAN DEEP LEARNING 101 [[website]](http://www.cs.ox.ac.uk/people/yarin.gal/website/bdl101/)
- MIT 6.S191: Evidential Deep Learning and Uncertainty (2021) [[Youtube]](https://www.youtube.com/watch?v=toTcf7tZK8c)

# Other-resources
Awesome conformal prediction [[github]](https://github.com/valeman/awesome-conformal-prediction)

Uncertainty Quantification in Deep Learning [[github]](https://github.com/ahmedmalaa/deep-learning-uncertainty)