# Awesome Uncertainty in Deep learning

<div align="center">

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

</div>

This repo is a collection of *awesome* papers, codes, books, and blogs about Uncertainty and Deep learning. 

:star: Feel free to star and fork. :star:

If you think we missed a paper, please open a pull request or send a message on the corresponding [GitHub discussion](https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning/discussions). Tell us where the article was published and when, and send us GitHub and ArXiv links if they are available.

We are also open to any ideas for improvements!

<h2>
Table of Contents
</h2>

- [Awesome Uncertainty in Deep learning](#awesome-uncertainty-in-deep-learning)
- [Papers](#papers)
  - [Surveys](#surveys)
  - [Theory](#theory)
  - [Bayesian-Methods](#bayesian-methods)
  - [Ensemble-Methods](#ensemble-methods)
  - [Sampling/Dropout-based-Methods](#samplingdropout-based-methods)
  - [Post-hoc-Methods/Auxiliary-Networks](#post-hoc-methodsauxiliary-networks)
  - [Data-augmentation/Generation-based-methods](#data-augmentationgeneration-based-methods)
  - [Output-Space-Modeling/Evidential-deep-learning](#output-space-modelingevidential-deep-learning)
  - [Deterministic-Uncertainty-Methods](#deterministic-uncertainty-methods)
  - [Quantile-Regression/Predicted-Intervals](#quantile-regressionpredicted-intervals)
  - [Conformal Predictions](#conformal-predictions)
  - [Calibration/Evaluation-Metrics](#calibrationevaluation-metrics)
  - [Misclassification Detection \& Selective Classification](#misclassification-detection--selective-classification)
  - [Applications](#applications)
    - [Classification and Semantic-Segmentation](#classification-and-semantic-segmentation)
    - [Regression](#regression)
    - [Anomaly-detection and Out-of-Distribution-Detection](#anomaly-detection-and-out-of-distribution-detection)
    - [Object detection](#object-detection)
    - [Domain adaptation](#domain-adaptation)
    - [Semi-supervised](#semi-supervised)
    - [Natural Language Processing](#natural-language-processing)
    - [Others](#others)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Libraries](#libraries)
  - [Python](#python)
  - [PyTorch](#pytorch)
  - [JAX](#jax)
  - [TensorFlow](#tensorflow)
- [Lectures and tutorials](#lectures-and-tutorials)
- [Books](#books)
- [Other Resources](#other-resources)

# Papers

## Surveys

**Conference**

- Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks [[NeurIPS2024](<https://arxiv.org/abs/2402.19460>) - [[PyTorch]](<https://github.com/bmucsanyi/untangle>)
- A Comparison of Uncertainty Estimation Approaches in Deep Learning Components for Autonomous Vehicle Applications [[AISafety Workshop 2020]](<https://arxiv.org/abs/2006.15172>)

**Journal**

- A survey of uncertainty in deep neural networks [[Artificial Intelligence Review 2023]](<https://arxiv.org/abs/2107.03342>) - [[GitHub]](<https://github.com/JakobCode/UncertaintyInNeuralNetworks_Resources>) 
- Prior and Posterior Networks: A Survey on Evidential Deep Learning Methods For Uncertainty Estimation [[TMLR2023]](<https://arxiv.org/abs/2110.03051>)
- A Survey on Uncertainty Estimation in Deep Learning Classification Systems from a Bayesian Perspective [[ACM2021]](<https://dl.acm.org/doi/pdf/10.1145/3477140?casa_token=6fozCYTovlIAAAAA:t5vcjuXCMem1b8iFwaMG4o_YJHTe0wArLtoy9KCbL8Cow0aGEoxSiJans2Kzpm2FSKOg-4ZCDkBa>)
- Ensemble deep learning: A review [[Engineering Applications of AI 2021]](<https://arxiv.org/abs/2104.02395>)
- A review of uncertainty quantification in deep learning: Techniques, applications and challenges [[Information Fusion 2021]](<https://www.sciencedirect.com/science/article/pii/S1566253521001081>)
- Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods [[Machine Learning 2021]](<https://link.springer.com/article/10.1007/s10994-021-05946-3>)
- Predictive inference with the jackknife+ [[The Annals of Statistics 2021]](<https://arxiv.org/abs/1905.02928>)
- Uncertainty in big data analytics: survey, opportunities, and challenges [[Journal of Big Data 2019]](<https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0206-3?cv=1>)

**Arxiv**

- A System-Level View on Out-of-Distribution Data in Robotics [[arXiv2022]](<https://arxiv.org/abs/2212.14020>)
- A Survey on Uncertainty Reasoning and Quantification for Decision Making: Belief Theory Meets Deep Learning [[arXiv2022]](<https://arxiv.org/abs/2206.05675>)

## Theory

**Conference**

- A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods [[NeurIPS2023]](<https://arxiv.org/pdf/2305.15027>)
- Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning [[ICLR2023]](<https://arxiv.org/pdf/2012.09816.pdf>)
- Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask? [[ICLR2023]](<https://arxiv.org/pdf/2210.03044.pdf>)
- Probabilistic Contrastive Learning Recovers the Correct Aleatoric Uncertainty of Ambiguous Inputs [[ICML2023]](<https://arxiv.org/pdf/2302.02865.pdf>) - [[PyTorch]](<https://github.com/mkirchhof/Probabilistic_Contrastive_Learning>)
- On Second-Order Scoring Rules for Epistemic Uncertainty Quantification [[ICML2023]](<https://arxiv.org/pdf/2301.12736.pdf>)
- Neural Variational Gradient Descent [[AABI2022]](<https://openreview.net/forum?id=oG0vTBw58ic>)
- Top-label calibration and multiclass-to-binary reductions [[ICLR2022]](<https://openreview.net/forum?id=WqoBaaPHS->)
- Bayesian Model Selection, the Marginal Likelihood, and Generalization [[ICML2022]](<https://arxiv.org/abs/2202.11678>)
- With malice towards none: Assessing uncertainty via equalized coverage [[AIES 2021]](<https://arxiv.org/abs/1908.05428>)
- Uncertainty in Gradient Boosting via Ensembles [[ICLR2021]](<https://arxiv.org/abs/2006.10562>) - [[PyTorch]](<https://github.com/yandex-research/GBDT-uncertainty>)
- Repulsive Deep Ensembles are Bayesian [[NeurIPS2021]](<https://arxiv.org/abs/2106.11642>) - [[PyTorch]](<https://github.com/ratschlab/repulsive_ensembles>)
- Bayesian Optimization with High-Dimensional Outputs [[NeurIPS2021]](<https://arxiv.org/abs/2106.12997>)
- Residual Pathway Priors for Soft Equivariance Constraints [[NeurIPS2021]](<https://arxiv.org/abs/2112.01388>)
- Dangers of Bayesian Model Averaging under Covariate Shift [[NeurIPS2021]](<https://arxiv.org/abs/2106.11905>) - [[TensorFlow]](<https://github.com/izmailovpavel/bnn_covariate_shift>)
- A Mathematical Analysis of Learning Loss for Active Learning in Regression [[CVPR Workshop2021]](<https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.html>)
- Why Are Bootstrapped Deep Ensembles Not Better? [[NeurIPS Workshop]](<https://openreview.net/forum?id=dTCir0ceyv0>)
- Deep Convolutional Networks as shallow Gaussian Processes [[ICLR2019]](<https://arxiv.org/abs/1808.05587>)
- On the accuracy of influence functions for measuring group effects [[NeurIPS2018]](<https://proceedings.neurips.cc/paper/2019/hash/a78482ce76496fcf49085f2190e675b4-Abstract.html>)
- To Trust Or Not To Trust A Classifier [[NeurIPS2018]](<https://arxiv.org/abs/1805.11783>) - [[Python]](<https://github.com/google/TrustScore>)
- Understanding Measures of Uncertainty for Adversarial Example Detection [[UAI2018]](<https://arxiv.org/abs/1803.08533>)

**Journal**

- Martingale posterior distributions [[Royal Statistical Society Series B]](<https://arxiv.org/abs/2103.15671>)
- A Unified Theory of Diversity in Ensemble Learning [[JMLR2023]](<https://jmlr.org/papers/volume24/23-0041/23-0041.pdf>)
- Multivariate Uncertainty in Deep Learning [[TNNLS2021]](<https://arxiv.org/abs/1910.14215>)
- A General Framework for Uncertainty Estimation in Deep Learning [[RAL2020]](<https://arxiv.org/abs/1907.06890>)
- Adaptive nonparametric confidence sets [[Ann. Statist. 2006]](<https://arxiv.org/abs/math/0605473>)

**Arxiv**

- Ensembles for Uncertainty Estimation: Benefits of Prior Functions and Bootstrapping [[arXiv2022]](<https://arxiv.org/pdf/2206.03633.pdf>)
- Efficient Gaussian Neural Processes for Regression [[arXiv2021]](<https://arxiv.org/abs/2108.09676>)
- Dense Uncertainty Estimation [[arXiv2021]](<https://arxiv.org/abs/2110.06427>) - [[PyTorch]](<https://github.com/JingZhang617/UncertaintyEstimation>)
- A higher-order swiss army infinitesimal jackknife [[arXiv2019]](<https://arxiv.org/abs/1907.12116>)

## Bayesian-Methods

**Conference**

- Training Bayesian Neural Networks with Sparse Subspace Variational Inference [[ICLR2024]](<https://arxiv.org/abs/2402.11025>)
- Variational Bayesian Last Layers [[ICLR2024]](https://arxiv.org/abs/2404.11599)
- A Symmetry-Aware Exploration of Bayesian Neural Network Posteriors [[ICLR2024]](<https://arxiv.org/abs/2310.08287>)
- Uncertainty-aware Unsupervised Video Hashing [[AISTATS2023]](<https://proceedings.mlr.press/v206/wang23i.html>) - [[PyTorch]](<https://github.com/wangyucheng1234/BerVAE>)
- Gradient-based Uncertainty Attribution for Explainable Bayesian Deep Learning [[CVPR2023]](<https://arxiv.org/abs/2304.04824>)
- Robustness to corruption in pre-trained Bayesian neural networks [[ICLR2023]](<https://arxiv.org/pdf/2206.12361.pdf>)
- Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift [[NeurIPS2023]](<https://arxiv.org/abs/2306.12306>) - [[PyTorch]](<https://github.com/Feuermagier/Beyond_Deep_Ensembles>)
- Transformers Can Do Bayesian Inference [[ICLR2022]](<https://arxiv.org/abs/2112.10510>) - [[PyTorch]](<https://github.com/automl/PFNs?tab=readme-ov-file>)
- Uncertainty Estimation for Multi-view Data: The Power of Seeing the Whole Picture [[NeurIPS2022]](<https://arxiv.org/abs/2210.02676>)
- On Batch Normalisation for Approximate Bayesian Inference [[AABI2021]](<https://openreview.net/pdf?id=SH2tfpm_0LE>)
- Activation-level uncertainty in deep neural networks [[ICLR2021]](<https://openreview.net/forum?id=UvBPbpvHRj->)
- Laplace Redux – Effortless Bayesian Deep Learning [[NeurIPS2021]](<https://arxiv.org/abs/2106.14806>) - [[PyTorch]](<https://github.com/AlexImmer/Laplace>)
- On the Effects of Quantisation on Model Uncertainty in Bayesian Neural Networks [[UAI2021]](<https://arxiv.org/abs/2102.11062>)
- Learnable uncertainty under Laplace approximations [[UAI2021]](<https://proceedings.mlr.press/v161/kristiadi21a.html>)
- Bayesian Neural Networks with Soft Evidence [[ICML Workshop2021]](<https://arxiv.org/abs/2010.09570>) - [[PyTorch]](<https://github.com/edwardyu/soft-evidence-bnn>)
- TRADI: Tracking deep neural network weight distributions for uncertainty estimation [[ECCV2020]](<https://arxiv.org/abs/1912.11316>) - [[PyTorch]](<https://github.com/giannifranchi/TRADI_Tracking_DNN_weights>)
- How Good is the Bayes Posterior in Deep Neural Networks Really? [[ICML2020]](<http://proceedings.mlr.press/v119/wenzel20a.html>)
- Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors [[ICML2020]](<http://proceedings.mlr.press/v119/dusenberry20a/dusenberry20a.pdf>) - [[TensorFlow]](<https://github.com/google/edward2>)
- Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks [[ICML2020]](<http://proceedings.mlr.press/v119/kristiadi20a/kristiadi20a.pdf>) - [[PyTorch]](<https://github.com/AlexImmer/Laplace>)
- Bayesian Deep Learning and a Probabilistic Perspective of Generalization [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/file/322f62469c5e3c7dc3e58f5a4d1ea399-Paper.pdf>)
- A Simple Baseline for Bayesian Uncertainty in Deep Learning [[NeurIPS2019]](<https://arxiv.org/abs/1902.02476>) - [[PyTorch]](<https://github.com/wjmaddox/swa_gaussian>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)
- Bayesian Uncertainty Estimation for Batch Normalized Deep Networks [[ICML2018]](<http://proceedings.mlr.press/v80/teye18a.html>) - [[TensorFlow]](<https://github.com/icml-mcbn/mcbn>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](<https://github.com/ezjong/lightprobnets>) - [[PyTorch]](<https://github.com/ezjong/lightprobnets>)
- A Scalable Laplace Approximation for Neural Networks [[ICLR2018]](<https://openreview.net/pdf?id=Skdvd2xAZ>) - [[Theano]](<https://github.com/BB-UCL/Lasagne>)
- Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning [[ICML2018]](<http://proceedings.mlr.press/v80/depeweg18a.html>)
- Weight Uncertainty in Neural Networks [[ICML2015]](<https://proceedings.mlr.press/v37/blundell15.html>)

**Journal**

- Hashing with Uncertainty Quantification via Sampling-based Hypothesis Testing [[TMLR2024]](<https://openreview.net/forum?id=cc4v6v310f>) - [[PyTorch]](<https://github.com/QianLab/HashUQ>)
- Analytically Tractable Hidden-States Inference in Bayesian Neural Networks [[JMLR2024]](<https://jmlr.org/papers/v23/21-0758.html>)
- Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification [[TPAMI2023]](<https://arxiv.org/abs/2012.02818>) - [[PyTorch]](<https://github.com/giannifranchi/LP_BNN>)
- Bayesian modeling of uncertainty in low-level vision [[IJCV1990]](<https://link.springer.com/article/10.1007%2FBF00126502>)

**Arxiv**

- Density Uncertainty Layers for Reliable Uncertainty Estimation [[arXiv2023]](<https://arxiv.org/abs/2306.12497>)

## Ensemble-Methods

**Conference**
- Divergent Ensemble Networks: Enhancing Uncertainty Estimation with Shared Representations and Independent Branching [[ICJI2024]](https://arxiv.org/abs/2412.01193)
- Input-gradient space particle inference for neural network ensembles [[ICLR2024]](<https://arxiv.org/abs/2306.02775>)
- Fast Ensembling with Diffusion Schrödinger Bridge [[ICLR2024]](<https://arxiv.org/abs/2404.15814>)
- Pathologies of Predictive Diversity in Deep Ensembles [[ICLR2024]](<https://arxiv.org/abs/2302.00704>)
- Model Ratatouille: Recycling Diverse Models for Out-of-Distribution Generalization [[ICML2023]](<https://arxiv.org/pdf/2212.10445.pdf>)
- Bayesian Posterior Approximation With Stochastic Ensembles [[CVPR2023]](<https://openaccess.thecvf.com/content/CVPR2023/papers/Balabanov_Bayesian_Posterior_Approximation_With_Stochastic_Ensembles_CVPR_2023_paper.pdf>)
- Normalizing Flow Ensembles for Rich Aleatoric and Epistemic Uncertainty Modeling [[AAAI2023]](<https://arxiv.org/abs/2302.01312>)
- Window-Based Early-Exit Cascades for Uncertainty Estimation: When Deep Ensembles are More Efficient than Single Models [[ICCV2023]](<https://arxiv.org/abs/2303.08010>) - [[PyTorch]](<https://github.com/guoxoug/window-early-exit>)
- Weighted Ensemble Self-Supervised Learning [[ICLR2023]](<https://arxiv.org/pdf/2211.09981.pdf>)
- Agree to Disagree: Diversity through Disagreement for Better Transferability [[ICLR2023]](<https://arxiv.org/pdf/2202.04414.pdf>) - [[PyTorch]](<https://github.com/mpagli/Agree-to-Disagree>)
- Packed-Ensembles for Efficient Uncertainty Estimation [[ICLR2023]](<https://arxiv.org/abs/2210.09184>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)
- Sub-Ensembles for Fast Uncertainty Estimation in Neural Networks [[ICCV Workshop2023]](<https://openaccess.thecvf.com/content/ICCV2023W/LXCV/papers/Valdenegro-Toro_Sub-Ensembles_for_Fast_Uncertainty_Estimation_in_Neural_Networks_ICCVW_2023_paper.pdf>)
- Prune and Tune Ensembles: Low-Cost Ensemble Learning With Sparse Independent Subnetworks [[AAAI2022]](<https://arxiv.org/abs/2202.11782>)
- Deep Ensembles Work, But Are They Necessary? [[NeurIPS2022]](<https://arxiv.org/abs/2202.06985>)
- FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation [[NeurIPS2022]](<https://arxiv.org/abs/2206.00050>)
- Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity [[ICLR2022]](<https://arxiv.org/abs/2106.14568>) - [[PyTorch]](<https://github.com/VITA-Group/FreeTickets>)
- On the Usefulness of Deep Ensemble Diversity for Out-of-Distribution Detection [[ECCV Workshop2022]](<https://arxiv.org/abs/2207.07517>)
- Masksembles for Uncertainty Estimation [[CVPR2021]](<https://nikitadurasov.github.io/projects/masksembles/>) - [[PyTorch/TensorFlow]](<https://github.com/nikitadurasov/masksembles>)
- Robustness via Cross-Domain Ensembles [[ICCV2021]](<https://arxiv.org/abs/2103.10919>) - [[PyTorch]](<https://github.com/EPFL-VILAB/XDEnsembles>)
- Uncertainty in Gradient Boosting via Ensembles [[ICLR2021]](<https://arxiv.org/abs/2006.10562>) - [[PyTorch]](<https://github.com/yandex-research/GBDT-uncertainty>)
- Uncertainty Quantification and Deep Ensembles [[NeurIPS2021]](<https://openreview.net/forum?id=wg_kD_nyAF>)
- Maximizing Overall Diversity for Improved Uncertainty Estimates in Deep Ensembles [[AAAI2020]](<https://ojs.aaai.org/index.php/AAAI/article/view/5849>)
- Uncertainty in Neural Networks: Approximately Bayesian Ensembling [[AISTATS2020]](<https://arxiv.org/abs/1810.05546>)
- Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning [[ICLR2020]](<https://arxiv.org/abs/2002.06470>) - [[PyTorch]](<https://github.com/SamsungLabs/pytorch-ensembles>)
- BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning [[ICLR2020]](<https://arxiv.org/abs/2002.06715>) - [[TensorFlow]](<https://github.com/google/edward2>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)
- Hyperparameter Ensembles for Robustness and Uncertainty Quantification [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/481fbfa59da2581098e841b7afc122f1-Abstract.html>)
- Bayesian Deep Ensembles via the Neural Tangent Kernel [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/0b1ec366924b26fc98fa7b71a9c249cf-Abstract.html>)
- Diversity with Cooperation: Ensemble Methods for Few-Shot Classification [[ICCV2019]](<https://arxiv.org/abs/1903.11341>)
- Accurate Uncertainty Estimation and Decomposition in Ensemble Learning [[NeurIPS2019]](<https://papers.nips.cc/paper/2019/hash/1cc8a8ea51cd0adddf5dab504a285915-Abstract.html>)
- High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach [[ICML2018]](<https://arxiv.org/abs/1802.07167>) - [[TensorFlow]](<https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals>)
- Snapshot Ensembles: Train 1, get M for free [[ICLR2017]](https://arxiv.org/abs/1704.00109) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)
- Simple and scalable predictive uncertainty estimation using deep ensembles [[NeurIPS2017]](<https://arxiv.org/abs/1612.01474>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)

**Journal**

- One Versus all for deep Neural Network for uncertainty (OVNNI) quantification [[IEEE Access2021]](<https://arxiv.org/abs/2006.00954>)

**Arxiv**

- Split-Ensemble: Efficient OOD-aware Ensemble via Task and Model Splitting [[arXiv2023]](<https://arxiv.org/abs/2312.09148>)
- Deep Ensemble as a Gaussian Process Approximate Posterior [[arXiv2022]](<https://arxiv.org/abs/2205.00163>)
- Sequential Bayesian Neural Subnetwork Ensembles [[arXiv2022]](<https://arxiv.org/abs/2206.00794>)
- Confident Neural Network Regression with Bootstrapped Deep Ensembles [[arXiv2022]](<https://arxiv.org/abs/2202.10903>) - [[TensorFlow]](<https://github.com/LaurensSluyterman/Bootstrapped_Deep_Ensembles>)
- Dense Uncertainty Estimation via an Ensemble-based Conditional Latent Variable Model [[arXiv2021]](<https://arxiv.org/abs/2111.11055>)
- Deep Ensembles: A Loss Landscape Perspective [[arXiv2019]](<https://arxiv.org/abs/1912.02757>)
- Checkpoint ensembles: Ensemble methods from a single training process [[arXiv2017]](<https://arxiv.org/abs/1710.03282>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)

## Sampling/Dropout-based-Methods

**Conference**

- Enabling Uncertainty Estimation in Iterative Neural Networks [[ICML2024]](<https://arxiv.org/pdf/2403.16732>) - [[Pytorch]](<https://github.com/cvlab-epfl/iter_unc>)
- Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models [[CVPR2024]](<https://arxiv.org/abs/2312.15297>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)
- Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate [[AAAI2022]](<https://arxiv.org/abs/1910.04858v3>)
- Efficient Bayesian Uncertainty Estimation for nnU-Net [[MICCAI2022]](<https://link.springer.com/chapter/10.1007/978-3-031-16452-1_51>)
- Dropout Sampling for Robust Object Detection in Open-Set Conditions [[ICRA2018]](<https://arxiv.org/abs/1710.06677>)
- Test-time data augmentation for estimation of heteroscedastic aleatoric uncertainty in deep neural networks [[MIDL2018]](<https://openreview.net/forum?id=rJZz-knjz>)
- Concrete Dropout [[NeurIPS2017]](<https://arxiv.org/abs/1705.07832>)
- Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning [[ICML2016]](<https://arxiv.org/abs/1506.02142>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)

**Journal**

- A General Framework for Uncertainty Estimation in Deep Learning [[Robotics and Automation Letters2020]](<https://arxiv.org/pdf/1907.06890.pdf>)

**Arxiv**

- SoftDropConnect (SDC) – Effective and Efficient Quantification of the Network Uncertainty in Deep MR Image Analysis [[arXiv2022]](<https://arxiv.org/abs/2201.08418>)

## Post-hoc-Methods/Auxiliary-Networks

**Conference**

- On the Limitations of Temperature Scaling for Distributions with Overlaps [[ICLR2024]](https://arxiv.org/abs/2306.00740)
- Post-hoc Uncertainty Learning using a Dirichlet Meta-Model [[AAAI2023]](<https://arxiv.org/abs/2212.07359>) - [[PyTorch]](<https://github.com/maohaos2/PosthocUQ>)
- ProbVLM: Probabilistic Adapter for Frozen Vision-Language Models [[ICCV2023]](<https://openaccess.thecvf.com/content/ICCV2023/html/Upadhyay_ProbVLM_Probabilistic_Adapter_for_Frozen_Vison-Language_Models_ICCV_2023_paper.html>)
- Out-of-Distribution Detection for Monocular Depth Estimation [[ICCV2023]](<https://arxiv.org/abs/2308.06072>)
- Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model [[AAAI2022]](<https://ojs.aaai.org/index.php/AAAI/article/view/20773>)
- Learning Structured Gaussians to Approximate Deep Ensembles [[CVPR2022]](<https://arxiv.org/abs/2203.15485>)
- Improving the reliability for confidence estimation [[ECCV2022]](<https://arxiv.org/abs/2210.06776>)
- Gradient-based Uncertainty for Monocular Depth Estimation [[ECCV2022]](<https://arxiv.org/abs/2208.02005>) - [[PyTorch]](<https://github.com/jhornauer/GrUMoDepth>)
- BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks [[ECCV2022]](<https://arxiv.org/abs/2207.06873>) - [[PyTorch]](<https://github.com/ExplainableML/BayesCap>)
- Learning Uncertainty For Safety-Oriented Semantic Segmentation In Autonomous Driving [[ICIP2022]](<https://arxiv.org/abs/2105.13688>)
- SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](<https://arxiv.org/abs/2104.02395>) - [[PyTorch]](<https://github.com/xuanlongORZ/SLURP_uncertainty_estimate>)
- Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks in Semantic Segmentation [[ICCV2021]](<https://arxiv.org/abs/2108.01634>) - [[PyTorch]](<https://github.com/valeoai/obsnet>)
- Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](<https://arxiv.org/abs/2002.05582>)
- A Mathematical Analysis of Learning Loss for Active Learning in Regression [[CVPR Workshop2021]](<https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.html>)
- Real-time uncertainty estimation in computer vision via uncertainty-aware distribution distillation [[WACV2021]](<https://arxiv.org/abs/2007.15857>)
- On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](<https://arxiv.org/abs/2005.06209>) - [[PyTorch]](<https://github.com/mattpoggi/mono-uncertainty>)
- Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel [[ICLR2020]](<https://arxiv.org/abs/1906.00588>) - [[TensorFlow]](<https://github.com/cognizant-ai-labs/rio-paper>)
- Gradients as a Measure of Uncertainty in Neural Networks [[ICIP2020]](<https://arxiv.org/abs/2008.08030>)
- Learning Loss for Test-Time Augmentation [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html>)
- Learning loss for active learning [[CVPR2019]](<https://arxiv.org/abs/1905.03677>) - [[PyTorch]](<https://github.com/Mephisto405/Learning-Loss-for-Active-Learning>) (unofficial codes)
- Addressing failure prediction by learning model confidence [[NeurIPS2019]](<https://papers.NeurIPS.cc/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf>) - [[PyTorch]](<https://github.com/valeoai/ConfidNet>)
- Structured Uncertainty Prediction Networks [[CVPR2018]](<https://arxiv.org/abs/1802.07079>) - [[TensorFlow]](<https://github.com/Era-Dorta/tf_mvg>)
- Classification uncertainty of deep neural networks based on gradient information [[IAPR Workshop2018]](<https://arxiv.org/abs/1805.08440>)

**Journal**

- Towards More Reliable Confidence Estimation [[TPAMI2023]](<https://ieeexplore.ieee.org/abstract/document/10172026/>)
- Confidence Estimation via Auxiliary Models [[TPAMI2021]](<https://arxiv.org/abs/2012.06508>)

**Arxiv**

- Instance-Aware Observer Network for Out-of-Distribution Object Segmentation [[arXiv2022]](<https://arxiv.org/abs/2207.08782>)
- DEUP: Direct Epistemic Uncertainty Prediction [[arXiv2020]](<https://arxiv.org/abs/2102.08501>)
- Learning Confidence for Out-of-Distribution Detection in Neural Networks [[arXiv2018]](<https://arxiv.org/abs/1802.04865>)

## Data-augmentation/Generation-based-methods

**Conference**

- Posterior Uncertainty Quantification in Neural Networks using Data Augmentation [[AISTATS2024]](<https://arxiv.org/abs/2403.12729>)
- Learning to Generate Training Datasets for Robust Semantic Segmentation [[WACV2024]](<https://arxiv.org/abs/2308.02535>)
- OpenMix: Exploring Outlier Samples for Misclassification Detection [[CVPR2023]](<https://arxiv.org/abs/2303.17093>) - [[PyTorch]](<https://github.com/Impression2805/OpenMix>)
- On the Pitfall of Mixup for Uncertainty Calibration [[CVPR2023]](<https://openaccess.thecvf.com/content/CVPR2023/html/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.html>)
- Diverse, Global and Amortised Counterfactual Explanations for Uncertainty Estimates [[AAAI2022]](<https://arxiv.org/abs/2112.02646>)
- Out-of-distribution Detection with Implicit Outlier Transformation [[ICLR2023]](<https://arxiv.org/abs/2303.05033>) - [[PyTorch]](<https://github.com/qizhouwang/doe>)
- PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures [[CVPR2022]](<https://arxiv.org/abs/2112.05135>)
- RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy & Out-of-Distribution Robustness [[NeurIPS2022]](<https://arxiv.org/abs/2206.14502>) - [[PyTorch]](<https://github.com/francescopinto/regmixup>)
- Towards efficient feature sharing in MIMO architectures [[CVPR Workshop2022]](<https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Sun_Towards_Efficient_Feature_Sharing_in_MIMO_Architectures_CVPRW_2022_paper.html>)
- Robust Semantic Segmentation with Superpixel-Mix [[BMVC2021]](<https://arxiv.org/abs/2108.00968>) - [[PyTorch]](<https://github.com/giannifranchi/deeplabv3-superpixelmix>)
- MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks [[ICCV2021]](<https://arxiv.org/abs/2103.06132>) - [[PyTorch]](<https://github.com/alexrame/mixmo-pytorch>)
- Training independent subnetworks for robust prediction [[ICLR2021]](<https://arxiv.org/abs/2010.06610>)
- Regularizing Variational Autoencoder with Diversity and Uncertainty Awareness [[IJCAI2021]](<https://arxiv.org/abs/2110.12381>) - [[PyTorch]](<https://github.com/smilesdzgk/du-vae>)
- Uncertainty-aware GAN with Adaptive Loss for Robust MRI Image Enhancement  [[ICCV Workshop2021]](<https://arxiv.org/pdf/2110.03343.pdf>)
- Uncertainty-Aware Deep Classifiers using Generative Models [[AAAI2020]](<https://arxiv.org/abs/2006.04183>)
- Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [[ECCV2020]](<https://arxiv.org/abs/2003.08440>) - [[PyTorch]](<https://github.com/YingdaXia/SynthCP>)
- Detecting the Unexpected via Image Resynthesis [[ICCV2019]](<https://arxiv.org/abs/1904.07595>) - [[PyTorch]](<https://github.com/cvlab-epfl/detecting-the-unexpected>)
- Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning [[ICML2020]](<http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf>)
- Deep Anomaly Detection with Outlier Exposure [[ICLR2019]](<https://arxiv.org/pdf/1812.04606.pdf>)
- On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks [[NeurIPS2019]](<https://arxiv.org/abs/1905.11001>)

**Arxiv**

- Reliability in Semantic Segmentation: Can We Use Synthetic Data? [[arXiv2023]](<https://arxiv.org/pdf/2312.09231.pdf>)
- Quantifying uncertainty with GAN-based priors [[arXiv2019]](<https://openreview.net/forum?id=HyeAPeBFwS>) - [[TensorFlow]](<https://github.com/dhruvpatel108/GANPriors>)

## Output-Space-Modeling/Evidential-deep-learning

**Conference**

- Hyper-opinion Evidential Deep Learning for Out-of-Distribution Detection [[NeurIPS2024]](<https://openreview.net/forum?id=Te8vI2wGTh&referrer=%5Bthe%20profile%20of%20Yufei%20Chen%5D(%2Fprofile%3Fid%3D~Yufei_Chen1)>)
- Hyper Evidential Deep Learning to Quantify Composite Classification Uncertainty [[ICLR2024]](https://arxiv.org/abs/2404.10980)
- The Evidence Contraction Issue in Deep Evidential Regression: Discussion and Solution [[AAAI2024]](<https://ojs.aaai.org/index.php/AAAI/article/view/30172>)
- Discretization-Induced Dirichlet Posterior for Robust Uncertainty Quantification on Regression [[AAAI2024]](<https://arxiv.org/abs/2308.09065>) - [[PyTorch]](<https://github.com/ENSTA-U2IS-AI/DIDO>)
- The Unreasonable Effectiveness of Deep Evidential Regression [[AAAI2023]](<https://arxiv.org/abs/2205.10060>) - [[PyTorch]](<https://github.com/pasteurlabs/unreasonable_effective_der>) - [[TorchUncertainty]](https://github.com/ENSTA-U2IS-AI/torch-uncertainty)
- Exploring and Exploiting Uncertainty for Incomplete Multi-View Classification [[CVPR2023]](https://arxiv.org/abs/2304.05165)
- Plausible Uncertainties for Human Pose Regression [[ICCV2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Bramlage_Plausible_Uncertainties_for_Human_Pose_Regression_ICCV_2023_paper.pdf) - [[PyTorch]](<https://github.com/biggzlar/plausible-uncertainties>)
- Uncertainty Estimation by Fisher Information-based Evidential Deep Learning [[ICML2023]](https://arxiv.org/pdf/2303.02045.pdf) - [[PyTorch]](<https://github.com/danruod/iedl>)
- Improving Evidential Deep Learning via Multi-task Learning [[AAAI2022]](<https://arxiv.org/abs/2112.09368>) - [[PyTorch]](https://github.com/deargen/MT-ENet)
- An Evidential Neural Network Model for Regression Based on Random Fuzzy Numbers [[BELIEF2022]](<https://arxiv.org/abs/2208.00647>)
- On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks [[ICLR2022]](<https://arxiv.org/abs/2203.09168>) - [[PyTorch]](https://github.com/martius-lab/beta-nll)
- Natural Posterior Network: Deep Bayesian Uncertainty for Exponential Family Distributions [[ICLR2022]](<https://arxiv.org/abs/2105.04471>) - [[PyTorch]](<https://github.com/borchero/natural-posterior-network>)
- Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation [[NeurIPS2022]](<https://openreview.net/pdf?id=epjxT_ARZW5>)
- Fast Predictive Uncertainty for Classification with Bayesian Deep Networks [[UAI2022]](<https://arxiv.org/abs/2003.01227>) - [[PyTorch]](<https://github.com/mariushobbhahn/LB_for_BNNs_official>)
- Evaluating robustness of predictive uncertainty estimation: Are Dirichlet-based models reliable? [[ICML2021]](<http://proceedings.mlr.press/v139/kopetzki21a/kopetzki21a.pdf>)
- Trustworthy multimodal regression with mixture of normal-inverse gamma distributions [[NeurIPS2021]](<https://arxiv.org/abs/2111.08456>)
- Misclassification Risk and Uncertainty Quantification in Deep Classifiers [[WACV2021]](<https://openaccess.thecvf.com/content/WACV2021/html/Sensoy_Misclassification_Risk_and_Uncertainty_Quantification_in_Deep_Classifiers_WACV_2021_paper.html>)
- Ensemble Distribution Distillation [[ICLR2020]](<https://arxiv.org/abs/1905.00076>)
- Conservative Uncertainty Estimation By Fitting Prior Networks [[ICLR2020]](<https://openreview.net/forum?id=BJlahxHYDS>)
- Being Bayesian about Categorical Probability [[ICML2020]](<https://arxiv.org/abs/2002.07965>) - [[PyTorch]](<https://github.com/tjoo512/belief-matching-framework>)
- Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts  [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/0eac690d7059a8de4b48e90f14510391-Abstract.html>) - [[PyTorch]](<https://github.com/sharpenb/Posterior-Network>)
- Deep Evidential Regression [[NeurIPS2020]](<https://arxiv.org/abs/1910.02600>) - [[TensorFlow]](<https://github.com/aamini/evidential-deep-learning>) - [[TorchUncertainty]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)
- Noise Contrastive Priors for Functional Uncertainty [[UAI2020]](<https://proceedings.mlr.press/v115/hafner20a.html>)
- Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples [[NeurIPS Workshop2020]](<https://arxiv.org/abs/2010.10474>)
- Uncertainty on Asynchronous Time Event Prediction [[NeurIPS2019]](<https://arxiv.org/abs/1911.05503>) - [[TensorFlow]](<https://github.com/sharpenb/Uncertainty-Event-Prediction>)
- Reverse KL-Divergence Training of Prior Networks: Improved Uncertainty and Adversarial Robustness [[NeurIPS2019]](<https://proceedings.neurips.cc/paper/2019/hash/7dd2ae7db7d18ee7c9425e38df1af5e2-Abstract.html>)
- Quantifying Classification Uncertainty using Regularized Evidential Neural Networks [[AAAI FSS2019]](<https://arxiv.org/abs/1910.06864>)
- Uncertainty estimates and multi-hypotheses networks for optical flow [[ECCV2018]](<https://arxiv.org/abs/1802.07095>) - [[TensorFlow]](<https://github.com/lmb-freiburg/netdef_models>)
- Evidential Deep Learning to Quantify Classification Uncertainty [[NeurIPS2018]](<https://arxiv.org/abs/1806.01768>) - [[PyTorch]](<https://github.com/dougbrion/pytorch-classification-uncertainty>)
- Predictive uncertainty estimation via prior networks [[NeurIPS2018]](<https://proceedings.neurips.cc/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html>)
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? [[NeurIPS2017]](<https://arxiv.org/abs/1703.04977>)
- Estimating the Mean and Variance of the Target Probability Distribution [[(ICNN1994)]](<https://ieeexplore.ieee.org/document/374138>)

**Journal**

- Prior and Posterior Networks: A Survey on Evidential Deep Learning Methods For Uncertainty Estimation [[TMLR2023]](<https://arxiv.org/abs/2110.03051>)
- Region-Based Evidential Deep Learning to Quantify Uncertainty and Improve Robustness of Brain Tumor Segmentation [[NCA2022]](<http://arxiv.org/abs/2208.06038>)
- An evidential classifier based on Dempster-Shafer theory and deep learning [[Neurocomputing2021]](<https://www.sciencedirect.com/science/article/pii/S0925231221004525>) - [[TensorFlow]](<https://github.com/tongzheng1992/E-CNN-classifier>)
- Evidential fully convolutional network for semantic segmentation [[AppliedIntelligence2021]](<https://link.springer.com/article/10.1007/s10489-021-02327-0>) - [[TensorFlow]](<https://github.com/tongzheng1992/E-FCN>)
- Information Aware max-norm Dirichlet networks for predictive uncertainty estimation [[NeuralNetworks2021]](<https://arxiv.org/abs/1910.04819#:~:text=Information%20Aware%20Max%2DNorm%20Dirichlet%20Networks%20for%20Predictive%20Uncertainty%20Estimation,-Theodoros%20Tsiligkaridis&text=Precise%20estimation%20of%20uncertainty%20in,prone%20to%20over%2Dconfident%20predictions>)
- A neural network classifier based on Dempster-Shafer theory [[IEEETransSMC2000]](<https://ieeexplore.ieee.org/abstract/document/833094/>)

**Arxiv**

- Evidential Uncertainty Quantification: A Variance-Based Perspective [[arXiv2023]](<https://arxiv.org/pdf/2311.11367.pdf>)
- Effective Uncertainty Estimation with Evidential Models for Open-World Recognition [[arXiv2022]](<https://openreview.net/pdf?id=NrB52z3eOTY>)
- Multivariate Deep Evidential Regression [[arXiv2022]](<https://arxiv.org/abs/2104.06135>)
- Regression Prior Networks [[arXiv2020]](<https://arxiv.org/abs/2006.11590>)
- A Variational Dirichlet Framework for Out-of-Distribution Detection [[arXiv2019]](<https://arxiv.org/abs/1811.07308>)
- Uncertainty estimation in deep learning with application to spoken language assessment [[PhDThesis2019]](<https://www.repository.cam.ac.uk/handle/1810/298857>)
- Inhibited softmax for uncertainty estimation in neural networks [[arXiv2018]](<https://arxiv.org/abs/1810.01861>)
- Quantifying Intrinsic Uncertainty in Classification via Deep Dirichlet Mixture Networks [[arXiv2018]](<https://arxiv.org/abs/1906.04450>)

## Deterministic-Uncertainty-Methods

**Conference**
- A Rate-Distortion View of Uncertainty Quantification [[ICML2024]](https://arxiv.org/abs/2406.10775) - [[Tensorflow]](https://github.com/ifiaposto/Distance_Aware_Bottleneck)
- Deep Deterministic Uncertainty: A Simple Baseline [[CVPR2023]](<https://arxiv.org/abs/2102.11582>) - [[PyTorch]](<https://github.com/omegafragger/DDU>)
- Gaussian Latent Representations for Uncertainty Estimation using Mahalanobis Distance in Deep Classifiers [[ICCV Workshop2023]](<https://openaccess.thecvf.com/content/ICCV2023W/UnCV/papers/Venkataramanan_Gaussian_Latent_Representations_for_Uncertainty_Estimation_Using_Mahalanobis_Distance_in_ICCVW_2023_paper.pdf>) - [[PyTorch]](<https://github.com/vaishwarya96/MAPLE-uncertainty-estimation>)
- A Simple and Explainable Method for Uncertainty Estimation using Attribute Prototype Networks [[ICCV Workshop2023]](<https://openaccess.thecvf.com/content/ICCV2023W/UnCV/papers/Zelenka_A_Simple_and_Explainable_Method_for_Uncertainty_Estimation_Using_Attribute_ICCVW_2023_paper.pdf>)
- Training, Architecture, and Prior for Deterministic Uncertainty Methods [[ICLR Workshop2023]](<https://arxiv.org/abs/2303.05796>) - [[PyTorch]](<https://github.com/orientino/dum-components>)
- Latent Discriminant deterministic Uncertainty [[ECCV2022]](<https://arxiv.org/abs/2207.10130>) - [[PyTorch]](<https://github.com/ENSTA-U2IS-AI/LDU>)
- On the Practicality of Deterministic Epistemic Uncertainty [[ICML2022]](<https://arxiv.org/abs/2107.00649>)
- Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression [[CoRR2021]](<https://arxiv.org/abs/2102.11409>)
- Uncertainty Estimation Using a Single Deep Deterministic Neural Network [[ICML2020]](<https://arxiv.org/abs/2003.02037>) - [[PyTorch]](<https://github.com/y0ast/deterministic-uncertainty-quantification>)
- Training normalizing flows with the information bottleneck for competitive generative classification [[NeurIPS2020]](<https://arxiv.org/abs/2001.06448>)
- Simple and principled uncertainty estimation with deterministic deep learning via distance awareness [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html>)
- Revisiting One-vs-All Classifiers for Predictive Uncertainty and Out-of-Distribution Detection in Neural Networks [[ICML Workshop2020]](<https://arxiv.org/abs/2007.05134>)
- Sampling-Free Epistemic Uncertainty Estimation Using Approximated Variance Propagation [[ICCV2019]](<https://openaccess.thecvf.com/content_ICCV_2019/html/Postels_Sampling-Free_Epistemic_Uncertainty_Estimation_Using_Approximated_Variance_Propagation_ICCV_2019_paper.html>) - [[PyTorch]](<https://github.com/janisgp/Sampling-free-Epistemic-Uncertainty>)
- Single-Model Uncertainties for Deep Learning [[NeurIPS2019]](<https://arxiv.org/abs/1811.00908>) - [[PyTorch]](<https://github.com/facebookresearch/SingleModelUncertainty/>)

**Journal**

- ZigZag: Universal Sampling-free Uncertainty Estimation Through Two-Step Inference [[TMLR2024]](<https://arxiv.org/pdf/2211.11435>) - [[Pytorch]](<https://github.com/cvlab-epfl/zigzag>)
- Density estimation in representation space [[EDSMLS2020]](<https://arxiv.org/abs/1908.07235>)

**Arxiv**

- The Hidden Uncertainty in a Neural Network’s Activations [[arXiv2020]](<https://arxiv.org/abs/2012.03082>)
- A simple framework for uncertainty in contrastive learning [[arXiv2020]](<https://arxiv.org/abs/2010.02038>)
- Distance-based Confidence Score for Neural Network Classifiers [[arXiv2017]](<https://arxiv.org/abs/1709.09844>)

## Quantile-Regression/Predicted-Intervals

**Conference**

- Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging [[ICML2022]](<https://arxiv.org/abs/2202.05265>) - [[PyTorch]](<https://github.com/aangelopoulos/im2im-uq>)
- Prediction Intervals: Split Normal Mixture from Quality-Driven Deep Ensembles [[UAI2020]](<http://proceedings.mlr.press/v124/saleh-salem20a.html>) - [[PyTorch]](<https://github.com/tarik/pi-snm-qde>)
- Classification with Valid and Adaptive Coverage [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html>)
- Single-Model Uncertainties for Deep Learning [[NeurIPS2019]](<https://arxiv.org/abs/1811.00908>) - [[PyTorch]](<https://github.com/facebookresearch/SingleModelUncertainty/>)
- High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach [[ICML2018]](<https://arxiv.org/abs/1802.07167>) - [[TensorFlow]](<https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals>)

**Journal**

- Scalable Uncertainty Quantification for Deep Operator Networks using Randomized Priors [[CMAME2022]](<https://arxiv.org/abs/2203.03048>)
- Exploring uncertainty in regression neural networks for construction of prediction intervals [[Neurocomputing2022]](<https://www.sciencedirect.com/science/article/abs/pii/S0925231222001102>)

**Arxiv**

- Interval Neural Networks: Uncertainty Scores [[arXiv2020]](<https://arxiv.org/abs/2003.11566>)
- Tight Prediction Intervals Using Expanded Interval Minimization [[arXiv2018]](<https://arxiv.org/abs/1806.11222>)

## Conformal Predictions

Awesome Conformal Prediction [[GitHub]](<https://github.com/valeman/awesome-conformal-prediction>)

<!-- **Conference**

- Testing for Outliers with Conformal p-values  [[Ann. Statist. 2023]](<https://arxiv.org/abs/2104.08279>) - [[Python]](<https://github.com/msesia/conditional-conformal-pvalues>)
- Uncertainty sets for image classifiers using conformal prediction [[ICLR2021]](https://arxiv.org/pdf/2009.14193.pdf) - [[GitHub]](https://github.com/aangelopoulos/conformal_classification)
- Conformal Prediction Under Covariate Shift [[NeurIPS2019]](<https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html>)
- Conformalized Quantile Regression [[NeurIPS2019]](<https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html>) -->

## Calibration/Evaluation-Metrics

**Conference**

- Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing [[ICLR2024]](<https://arxiv.org/abs/2309.12236>)
- Calibrating Transformers via Sparse Gaussian Processes [[ICLR2023]](<https://arxiv.org/abs/2303.02444>) - [[PyTorch]](<https://github.com/chenw20/sgpa>)
- Beyond calibration: estimating the grouping loss of modern neural networks [[ICLR2023]](<https://openreview.net/pdf?id=6w1k-IixnL8>) - [[Python]](<https://github.com/aperezlebel/beyond_calibration>)
- Dual Focal Loss for Calibration [[ICML 2023]](https://arxiv.org/abs/2305.13665)
- What Are Effective Labels for Augmented Data? Improving Calibration and Robustness with AutoLabel [[SaTML2023]](https://arxiv.org/abs/2302.11188)
- The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration [[CVPR2022]](<https://arxiv.org/abs/2111.15430>) - [[PyTorch]](<https://github.com/by-liu/mbls>)
- AdaFocal: Calibration-aware Adaptive Focal Loss [[NeurIPS2022]](https://arxiv.org/abs/2211.11838)
- Calibrating Deep Neural Networks by Pairwise Constraints [[CVPR2022]](<https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Calibrating_Deep_Neural_Networks_by_Pairwise_Constraints_CVPR_2022_paper.html>)
- Top-label calibration and multiclass-to-binary reductions [[ICLR2022]](<https://openreview.net/forum?id=WqoBaaPHS->)
- From label smoothing to label relaxation [[AAAI2021]](<https://www.aaai.org/AAAI21Papers/AAAI-2191.LienenJ.pdf>)
- Diagnostic Uncertainty Calibration: Towards Reliable Machine Predictions in Medical Domain [[AIStats2021]](https://arxiv.org/pdf/2007.01659)
- Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence [[NeurIPS2021]](<https://proceedings.neurips.cc/paper/2021/hash/61f3a6dbc9120ea78ef75544826c814e-Abstract.html>)
- Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification [[NeurIPS2021]](<https://arxiv.org/abs/2011.09588>)
- Soft Calibration Objectives for Neural Networks [[NeurIPS2021]](<https://proceedings.neurips.cc/paper_files/paper/2021/file/f8905bd3df64ace64a68e154ba72f24c-Paper.pdf>) - [[TensorFlow]](<https://github.com/google/uncertainty-baselines/tree/main/experimental/caltrain>)
- Confidence-Aware Learning for Deep Neural Networks [[ICML2020]](<https://arxiv.org/abs/2007.01458>) - [[PyTorch]](<https://github.com/daintlab/confidence-aware-learning>)
- Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning [[ICML2020]](<http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf>)
- Regularization via structural label smoothing [[ICML2020]](<https://proceedings.mlr.press/v108/li20e.html>)
- Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning [[MIDL2020]](<http://proceedings.mlr.press/v121/laves20a.html>) - [[PyTorch]](<https://github.com/mlaves/well-calibrated-regression-uncertainty>)
- Calibrating Deep Neural Networks using Focal Loss [[NeurIPS2020]](<https://arxiv.org/abs/2002.09437>) - [[PyTorch]](<https://github.com/torrvision/focal_calibration>)
- Stationary activations for uncertainty calibration in deep learning [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/18a411989b47ed75a60ac69d9da05aa5-Abstract.html>)
- Revisiting the evaluation of uncertainty estimation and its application to explore model complexity-uncertainty trade-off [[CVPR Workshop2020]](<https://openaccess.thecvf.com/content_CVPRW_2020/html/w1/Ding_Revisiting_the_Evaluation_of_Uncertainty_Estimation_and_Its_Application_to_CVPRW_2020_paper.html>)
- Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision [[CVPR Workshop2020]](<https://arxiv.org/abs/1906.01620>) - [[PyTorch]](<https://github.com/fregu856/evaluating_bdl>)
- Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers [[ICLR2019]](<https://arxiv.org/abs/1805.08206>)
- Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration [[NeurIPS2019]](<https://arxiv.org/pdf/1910.12656.pdf>) - [[GitHub]](<https://github.com/dirichletcal>)
- When does label smoothing help? [[NeurIPS2019]](<https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html>)
- Verified Uncertainty Calibration [[NeurIPS2019]](<https://papers.NeurIPS.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html>) - [[GitHub]](<https://github.com/p-lambda/verified_calibration>)
- Measuring Calibration in Deep Learning [[CVPR Workshop2019]](<https://arxiv.org/abs/1904.01685>)
- Accurate Uncertainties for Deep Learning Using Calibrated Regression [[ICML2018]](<https://arxiv.org/abs/1807.00263>)
- Generalized zero-shot learning with deep calibration network [[NeurIPS2018]](<https://proceedings.neurips.cc/paper/2018/hash/1587965fb4d4b5afe8428a4a024feb0d-Abstract.html>)
- On calibration of modern neural networks [[ICML2017]](<https://arxiv.org/abs/1706.04599>) - [[TorchUncertainty]](https://github.com/ENSTA-U2IS-AI/torch-uncertainty)
- On Fairness and Calibration [[NeurIPS2017]](<https://arxiv.org/abs/1709.02012>)
- Obtaining Well Calibrated Probabilities Using Bayesian Binning [[AAAI2015]](<https://ojs.aaai.org/index.php/AAAI/article/view/9602/9461>)

**Journal**

- Meta-Calibration: Learning of Model Calibration Using Differentiable Expected Calibration Error [[TMLR2023]](<https://arxiv.org/abs/2106.09613>) - [[PyTorch]](<https://github.com/ondrejbohdal/meta-calibration>)
- Evaluating and Calibrating Uncertainty Prediction in Regression Tasks [[Sensors2022]](<https://arxiv.org/abs/1905.11659>)
- Calibrated Prediction Intervals for Neural Network Regressors [[IEEE Access 2018]](<https://arxiv.org/abs/1803.09546>) - [[Python]](<https://github.com/cruvadom/Prediction_Intervals>)

**Arxiv**

- Towards Understanding Label Smoothing [[arXiv2020]](<https://arxiv.org/abs/2006.11653>)
- An Investigation of how Label Smoothing Affects Generalization [[arXiv2020]](<https://arxiv.org/abs/2010.12648>)
  
## Misclassification Detection & Selective Classification

- A Data-Driven Measure of Relative Uncertainty for Misclassification Detection [[ICLR2024]](https://arxiv.org/abs/2306.01710)
- Plugin estimators for selective classification with out-of-distribution detection [[ICLR2024]](https://arxiv.org/abs/2301.12386)
- SURE: SUrvey REcipes for building reliable and robust deep networks [[CVPR2024]](https://arxiv.org/abs/2403.00543) - [[PyTorch]](https://yutingli0606.github.io/SURE/)
- Augmenting Softmax Information for Selective Classification with Out-of-Distribution Data [[ACCV2022]](<https://openaccess.thecvf.com/content/ACCV2022/html/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.html>)
- Anomaly Detection via Reverse Distillation from One-Class Embedding [[CVPR2022]](<https://arxiv.org/abs/2201.10703>)
- Rethinking Confidence Calibration for Failure Prediction [[ECCV2022]](<https://link.springer.com/chapter/10.1007/978-3-031-19806-9_30>) - [[PyTorch]](<https://github.com/Impression2805/FMFP>)

## Uncertainty sources & Aleatoric and Epistemic Uncertainty Disentenglement

**Conference**

- Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks [[NeurIPS2024](<https://arxiv.org/abs/2402.19460>) - [[PyTorch]](<https://github.com/bmucsanyi/untangle>)

**ArXiv**

- Sources of Uncertainty in Machine Learning - A Statisticians’ View [[ArXiv2024]](<https://arxiv.org/pdf/2305.16703>)
- How disentangled are your classification uncertainties? [[ArXiv2024](<https://arxiv.org/abs/2408.12175>)

## Applications

### Classification and Semantic-Segmentation

**Conference**

- Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts [[ICLR2023]](<https://arxiv.org/abs/2212.07328>) - [[PyTorch]](<https://github.com/gaozhitong/mose-auseg>)
- Anytime Dense Prediction with Confidence Adaptivity [[ICLR2022]](<https://openreview.net/forum?id=kNKFOXleuC>) - [[PyTorch]](<https://github.com/liuzhuang13/anytime>)
- CRISP - Reliable Uncertainty Estimation for Medical Image Segmentation [[MICCAI2022]](<https://arxiv.org/abs/2206.07664>)
- TBraTS: Trusted Brain Tumor Segmentation [[MICCAI2022]](<https://arxiv.org/abs/2206.09309>) - [[PyTorch]](<https://github.com/cocofeat/tbrats>)
- Robust Semantic Segmentation with Superpixel-Mix [[BMVC2021]](<https://arxiv.org/abs/2108.00968>) - [[PyTorch]](<https://github.com/giannifranchi/deeplabv3-superpixelmix>)
- Deep Deterministic Uncertainty for Semantic Segmentation [[ICMLW2021]](<https://arxiv.org/abs/2111.00079>)
- DEAL: Difficulty-aware Active Learning for Semantic Segmentation [[ACCV2020]](<https://openaccess.thecvf.com/content/ACCV2020/html/Xie_DEAL_Difficulty-aware_Active_Learning_for_Semantic_Segmentation_ACCV_2020_paper.html>)
- Classification with Valid and Adaptive Coverage [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html>)
- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation [[ICCV2019]](<https://openaccess.thecvf.com/content_ICCV_2019/html/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.html>)
- Human Uncertainty Makes Classification More Robust [[ICCV2019]](<https://openaccess.thecvf.com/content_ICCV_2019/html/Peterson_Human_Uncertainty_Makes_Classification_More_Robust_ICCV_2019_paper.html>)
- Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation [[MICCAI2019]](<https://arxiv.org/abs/1806.05034>) - [[PyTorch]](<https://github.com/yulequan/UA-MT>)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](<https://arxiv.org/abs/1805.11327>) - [[PyTorch]](<https://github.com/ezjong/lightprobnets>)
- A Probabilistic U-Net for Segmentation of Ambiguous Images [[NeurIPS2018]](<https://arxiv.org/abs/1806.05034>) - [[PyTorch]](<https://github.com/stefanknegt/Probabilistic-Unet-Pytorch>)
- Evidential Deep Learning to Quantify Classification Uncertainty [[NeurIPS2018]](<https://arxiv.org/abs/1806.01768>) - [[PyTorch]](<https://github.com/dougbrion/pytorch-classification-uncertainty>)
- To Trust Or Not To Trust A Classifier [[NeurIPS2018]](<https://proceedings.neurips.cc/paper/2018/hash/7180cffd6a8e829dacfc2a31b3f72ece-Abstract.html>)
- Classification uncertainty of deep neural networks based on gradient information [[IAPR Workshop2018]](<https://arxiv.org/abs/1805.08440>)
- Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding [[BMVC2017]](<https://arxiv.org/abs/1511.02680>)

**Journal**

- Explainable machine learning in image classification models: An uncertainty quantification perspective." [[KnowledgeBased2022]](<https://www.sciencedirect.com/science/article/pii/S095070512200168X>)
- Region-Based Evidential Deep Learning to Quantify Uncertainty and Improve Robustness of Brain Tumor Segmentation [[NCA2022]](<https://arxiv.org/abs/2208.06038>)

**Arxiv**

- Leveraging Uncertainty Estimates to Improve Classifier Performance [[arXiv2023]](<https://arxiv.org/pdf/2311.11723.pdf>)
- Evaluating Bayesian Deep Learning Methods for Semantic Segmentation [[arXiv2018]](<https://arxiv.org/abs/1811.12709>)

### Regression

**Conference**

- Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation [[CVPR2023]](<https://arxiv.org/abs/2304.00152>) - [[PyTorch]](<https://github.com/lly00412/sednet>)
- Probabilistic MIMO U-Net: Efficient and Accurate Uncertainty Estimation for Pixel-wise Regression [[ICCV Workshop2023]](<https://arxiv.org/abs/2308.07477>) - [[PyTorch]](<https://github.com/antonbaumann/mimo-unet>)
- Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate [[AAAI2022]](<https://arxiv.org/abs/1910.04858v3>)
- Learning Structured Gaussians to Approximate Deep Ensembles [[CVPR2022]](<https://arxiv.org/abs/2203.15485>)
- Uncertainty Quantification in Depth Estimation via Constrained Ordinal Regression [[ECCV2022]](<https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620229.pdf>)
- On Monocular Depth Estimation and Uncertainty Quantification using Classification Approaches for Regression [[ICIP2022]](<https://arxiv.org/abs/2202.12369>)
- Anytime Dense Prediction with Confidence Adaptivity [[ICLR2022]](<https://openreview.net/forum?id=kNKFOXleuC>) - [[PyTorch]](<https://github.com/liuzhuang13/anytime>)
- Variational Depth Networks: Uncertainty-Aware Monocular Self-supervised Depth Estimation [[ECCV Workshop2022]](<https://link.springer.com/chapter/10.1007/978-3-031-25085-9_3>)
- SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](<https://arxiv.org/abs/2104.02395>) - [[PyTorch]](<https://github.com/xuanlongORZ/SLURP_uncertainty_estimate>)
- Robustness via Cross-Domain Ensembles [[ICCV2021]](<https://arxiv.org/abs/2103.10919>) - [[PyTorch]](<https://github.com/EPFL-VILAB/XDEnsembles>)
- Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](<https://arxiv.org/abs/2002.05582>)
- On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](<https://arxiv.org/abs/2005.06209>) - [[PyTorch]](<https://github.com/mattpoggi/mono-uncertainty>)
- Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel [[ICLR2020]](<https://arxiv.org/abs/1906.00588>) - [[TensorFlow]](<https://github.com/cognizant-ai-labs/rio-paper>)
- Fast Uncertainty Estimation for Deep Learning Based Optical Flow [[IROS2020]](<https://authors.library.caltech.edu/104758/>)
- Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning [[MIDL2020]](<http://proceedings.mlr.press/v121/laves20a.html>) - [[PyTorch]](<https://github.com/mlaves/well-calibrated-regression-uncertainty>)
- Deep Evidential Regression [[NeurIPS2020]](<https://arxiv.org/abs/1910.02600>) - [[TensorFlow]](<https://github.com/aamini/evidential-deep-learning>)
- Inferring Distributions Over Depth from a Single Image [[IROS2019]](<https://arxiv.org/abs/1912.06268>) - [[TensorFlow]](<https://github.com/gengshan-y/monodepth-uncertainty>)
- Multi-Task Learning based on Separable Formulation of Depth Estimation and its Uncertainty [[CVPR Workshop2019]](<https://openaccess.thecvf.com/content_CVPRW_2019/html/Uncertainty_and_Robustness_in_Deep_Visual_Learning/Asai_Multi-Task_Learning_based_on_Separable_Formulation_of_Depth_Estimation_and_CVPRW_2019_paper.html>)
- Lightweight Probabilistic Deep Networks [[CVPR2018]](<https://arxiv.org/abs/1805.11327>) - [[PyTorch]](<https://github.com/ezjong/lightprobnets>)
- Structured Uncertainty Prediction Networks [[CVPR2018]](<https://arxiv.org/abs/1802.07079>) - [[TensorFlow]](<https://github.com/Era-Dorta/tf_mvg>)
- Uncertainty estimates and multi-hypotheses networks for optical flow [[ECCV2018]](<https://arxiv.org/abs/1802.07095>) - [[TensorFlow]](<https://github.com/lmb-freiburg/netdef_models>)
- Accurate Uncertainties for Deep Learning Using Calibrated Regression [[ICML2018]](<https://arxiv.org/abs/1807.00263>)

**Journal**

- How Reliable is Your Regression Model's Uncertainty Under Real-World Distribution Shifts? [[TMLR2023]](<https://arxiv.org/abs/2302.03679>) - [[PyTorch]](<https://github.com/fregu856/regression_uncertainty>)
- Evaluating and Calibrating Uncertainty Prediction in Regression Tasks [[Sensors2022]](<https://arxiv.org/abs/1905.11659>)
- Exploring uncertainty in regression neural networks for construction of prediction intervals [[Neurocomputing2022]](<https://www.sciencedirect.com/science/article/abs/pii/S0925231222001102>)
- Wasserstein Dropout [[Machine Learning 2022]](<https://arxiv.org/abs/2012.12687>) - [[PyTorch]](<https://github.com/fraunhofer-iais/second-moment-loss>)
- Deep Distribution Regression [[Computational Statistics & Data Analysis2021]](<https://arxiv.org/abs/1903.06023>)
- Calibrated Prediction Intervals for Neural Network Regressors [[IEEE Access 2018]](<https://arxiv.org/abs/1803.09546>) - [[Python]](<https://github.com/cruvadom/Prediction_Intervals>)
- Learning a Confidence Measure for Optical Flow [[TPAMI2013]](<https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6261321&casa_token=fYVGhK2pa40AAAAA:XWJdS8zJ4JRw1brCIGiYpzEqMidXTTYVkcKTYnnhSl4ys5pUoHzHO6xsVeGZII9Ir1LAI_3YyfI&tag=1>)

**Arxiv**

- Understanding pathologies of deep heteroskedastic regression [[arxiv2024]](<https://arxiv.org/abs/2306.16717>)
- Measuring and Modeling Uncertainty Degree for Monocular Depth Estimation [[arXiv2023]](<https://arxiv.org/abs/2307.09929>)
- UncertaINR: Uncertainty Quantification of End-to-End Implicit Neural Representations for Computed Tomographaphy [[arXiv2022]](<https://arxiv.org/abs/2202.10847>)
- Efficient Gaussian Neural Processes for Regression [[arXiv2021]](<https://arxiv.org/abs/2108.09676>)

### Anomaly-detection and Out-of-Distribution-Detection

**Conference**

- Learning Transferable Negative Prompts for Out-of-Distribution Detection [[CVPR2024]](<https://arxiv.org/abs/2404.03248>) - [[PyTorch]](<https://github.com/mala-lab/negprompt>)
- Epistemic Uncertainty Quantification For Pre-trained Neural Networks [[CVPR2024]](<https://arxiv.org/abs/2404.10124>)
- NECO: NEural Collapse Based Out-of-distribution Detection [[ICLR2024]](<https://arxiv.org/abs/2310.06823>)
- When and How Does In-Distribution Label Help Out-of-Distribution Detection? [[ICML2024]](<https://arxiv.org/abs/2405.18635>) - [[PyTorch]](<https://github.com/deeplearning-wisc/id_label>)
- Anomaly Detection under Distribution Shift [[ICCV2023]](<https://arxiv.org/abs/2303.13845>) - [[PyTorch]](<https://github.com/mala-lab/ADShift>)
- Normalizing Flows for Human Pose Anomaly Detection [[ICCV2023]](https://orhir.github.io/STG_NF/) - [[PyTorch]](https://github.com/orhir/stg-nf)
- RbA: Segmenting Unknown Regions Rejected by All [[ICCV2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Nayal_RbA_Segmenting_Unknown_Regions_Rejected_by_All_ICCV_2023_paper.pdf) - [[PyTorch]](https://github.com/NazirNayal8/RbA)
- Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection [[CVPR2023]](<https://arxiv.org/abs/2303.10449>) - [[PyTorch]](<https://github.com/lufan31/et-ood>)
- Modeling the Distributional Uncertainty for Salient Object Detection Models [[CVPR2023]](https://npucvr.github.io/Distributional_uncer/) - [[PyTorch]](https://github.com/txynwpu/Distributional_uncertainty_SOD)
- SQUID: Deep Feature In-Painting for Unsupervised Anomaly Detection [[CVPR2023]](<https://arxiv.org/abs/2111.13495>) - [[PyTorch]](<https://github.com/tiangexiang/SQUID>)
- How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection? [[ICLR2023]](<https://arxiv.org/pdf/2203.04450.pdf>) - [[PyTorch]](<https://github.com/deeplearning-wisc/cider>)
- Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization [[ICLR2023]](<https://arxiv.org/pdf/2206.07837.pdf>)
- Can CNNs Be More Robust Than Transformers? [[ICLR2023]](<https://arxiv.org/pdf/2206.03452.pdf>)
- A framework for benchmarking class-out-of-distribution detection and its application to ImageNet [[ICLR2023]](<https://arxiv.org/pdf/2302.11893.pdf>)
- Extremely Simple Activation Shaping for Out-of-Distribution Detection [[ICLR2023]](<https://arxiv.org/abs/2209.09858>) - [[PyTorch]](<https://github.com/andrijazz/ash>)
- Quantification of Uncertainty with Adversarial Models [[NeurIPS2023]](<https://arxiv.org/abs/2307.03217>)
- The Robust Semantic Segmentation UNCV2023 Challenge Results [[ICCV Workshop2023]](https://arxiv.org/abs/2309.15478)
- Continual Evidential Deep Learning for Out-of-Distribution Detection [[ICCV Workshop2023]](https://openaccess.thecvf.com/content/ICCV2023W/VCL/html/Aguilar_Continual_Evidential_Deep_Learning_for_Out-of-Distribution_Detection_ICCVW_2023_paper.html)
- Far Away in the Deep Space: Nearest-Neighbor-Based Dense Out-of-Distribution Detection [[ICCV Workshop2023]](<https://arxiv.org/abs/2211.06660>)
- Gaussian Latent Representations for Uncertainty Estimation using Mahalanobis Distance in Deep Classifiers [[ICCV Workshop2023]](<https://arxiv.org/abs/2305.13849>)
- Calibrated Out-of-Distribution Detection with a Generic Representation [[ICCV Workshop2023]](<https://arxiv.org/abs/2303.13148>) - [[PyTorch]](<https://github.com/vojirt/grood>)
- Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model [[AAAI2022]](<https://ojs.aaai.org/index.php/AAAI/article/view/20773>)
- Towards Total Recall in Industrial Anomaly Detection [[CVPR2022]](<https://arxiv.org/abs/2106.08265>) - [[PyTorch]](<https://github.com/hcw-00/PatchCore_anomaly_detection>)
- POEM: Out-of-Distribution Detection with Posterior Sampling [[ICML2022]](<https://arxiv.org/abs/2206.13687>) - [[PyTorch]](<https://github.com/deeplearning-wisc/poem>)
- VOS: Learning What You Don't Know by Virtual Outlier Synthesis [[ICLR2022]](<https://arxiv.org/abs/2202.01197>) - [[PyTorch]](<https://github.com/deeplearning-wisc/vos>)
- Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection [[WACV2022]](<https://arxiv.org/abs/2110.02855>) - [[PyTorch]](<https://github.com/marco-rudolph/cs-flow>)
- Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces [[CVPR2021]](<https://openaccess.thecvf.com/content/CVPR2021/html/Zaeemzadeh_Out-of-Distribution_Detection_Using_Union_of_1-Dimensional_Subspaces_CVPR_2021_paper.html>) - [[PyTorch]](<https://github.com/zaeemzadeh/OOD>)
- NAS-OoD: Neural Architecture Search for Out-of-Distribution Generalization [[ICCV2021]](<https://arxiv.org/abs/2109.02038>)
- On the Importance of Gradients for Detecting Distributional Shifts in the Wild [[NeurIPS2021]](<https://arxiv.org/abs/2110.00218>)
- Exploring the Limits of Out-of-Distribution Detection [[NeurIPS2021]](<https://arxiv.org/abs/2106.03004>)
- Detecting out-of-distribution image without learning from out-of-distribution data. [[CVPR2020]](<https://openaccess.thecvf.com/content_CVPR_2020/html/Hsu_Generalized_ODIN_Detecting_Out-of-Distribution_Image_Without_Learning_From_Out-of-Distribution_Data_CVPR_2020_paper.html>)
- Learning Open Set Network with Discriminative Reciprocal Points [[ECCV2020]](<https://arxiv.org/abs/2011.00178>)
- Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [[ECCV2020]](<https://arxiv.org/abs/2003.08440>) - [[PyTorch]](<https://github.com/YingdaXia/SynthCP>)
- NADS: Neural Architecture Distribution Search for Uncertainty Awareness [[ICML2020]](<https://arxiv.org/abs/2006.06646>)
- PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization [[ICPR2020]](<https://arxiv.org/abs/2011.08785>) - [[PyTorch]](<https://github.com/openvinotoolkit/anomalib>)
- Energy-based Out-of-distribution Detection [[NeurIPS2020]](<https://arxiv.org/abs/2010.03759?context=cs>)
- Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples [[NeurIPS Workshop2020]](<https://arxiv.org/abs/2010.10474>)
- Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection [[ICCV2019]](<https://arxiv.org/abs/1904.02639>) - [[PyTorch]](<https://github.com/donggong1/memae-anomaly-detection>)
- Detecting the Unexpected via Image Resynthesis [[ICCV2019]](<https://arxiv.org/abs/1904.07595>) - [[PyTorch]](<https://github.com/cvlab-epfl/detecting-the-unexpected>)
- Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks [[ICLR2018]](<https://arxiv.org/abs/1706.02690>)
- A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks [[ICLR2017]](<https://arxiv.org/abs/1610.02136>) - [[TensorFlow]](<https://github.com/hendrycks/error-detection>)

**Journal**

- Generalized out-of-distribution detection: A survey [[IJCV2024]](<https://arxiv.org/abs/2110.11334>)
- Revisiting Confidence Estimation: Towards Reliable Failure Prediction [[TPAMI2024]](https://www.computer.org/csdl/journal/tp/5555/01/10356834/1SQHDHvGg9i) - [[PyTorch]](<https://github.com/Impression2805/FMFP>)
- One Versus all for deep Neural Network for uncertaInty (OVNNI) quantification [[IEEE Access2021]](<https://arxiv.org/abs/2006.00954>)

**Arxiv**

- Neuron Activation Coverage: Rethinking Out-of-distribution Detection and Generalization [[arXiv2023]](<https://arxiv.org/abs/2306.02879>) - [[PyTorch]](<https://github.com/bierone/ood_coverage>)
- A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection [[arXiv2021]](<https://arxiv.org/abs/2106.09022>)
- Do We Really Need to Learn Representations from In-domain Data for Outlier Detection? [[arXiv2021]](<https://arxiv.org/abs/2105.09270>)
- Frequentist uncertainty estimates for deep learning [[arXiv2018]](<http://bayesiandeeplearning.org/2018/papers/31.pdf>)

### Object detection

**Conference**

- Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection [[CVPR2023]](<https://arxiv.org/pdf/2303.14404.pdf>) - [[PyTorch]](<https://github.com/akhtarvision/bpc_calibration?tab=readme-ov-file>)
- Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection [[ECCV Workshop2022]](<https://arxiv.org/abs/2207.01242>) - [[PyTorch]](<https://github.com/EFS-OpenSource/calibration-framework>)
- Estimating and Evaluating Regression Predictive Uncertainty in Deep Object Detectors [[ICLR2021]](<https://openreview.net/forum?id=YLewtnvKgR7>) - [[PyTorch]](<https://github.com/asharakeh/probdet?tab=readme-ov-file>)
- Multivariate Confidence Calibration for Object Detection [[CVPR Workshop2020]](<https://arxiv.org/abs/2004.13546>) - [[PyTorch]](<https://github.com/EFS-OpenSource/calibration-framework>)
- Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving [[ICCV2019]](<https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.pdf>) - [[CUDA]](<https://github.com/jwchoi384/Gaussian_YOLOv3>) - [[PyTorch]](<https://github.com/motokimura/PyTorch_Gaussian_YOLOv3>) - [[Keras]](<https://github.com/xuannianz/keras-GaussianYOLOv3>)

### Domain adaptation

**Conference**

- Guiding Pseudo-labels with Uncertainty Estimation for Source-free Unsupervised Domain Adaptation [[CVPR2023]](<https://arxiv.org/abs/2303.03770>) - [[PyTorch]](https://github.com/mattialitrico/guiding-pseudo-labels-with-uncertainty-estimation-for-source-free-unsupervised-domain-adaptation)
- Uncertainty-guided Source-free
Domain Adaptation [[ECCV2022]](<https://arxiv.org/pdf/2208.07591.pdf>) - [[PyTorch]](<https://github.com/roysubhankar/uncertainty-sfda>)

### Semi-supervised

**Conference**

- Confidence Estimation Using Unlabeled Data [[ICLR2023]](<https://openreview.net/pdf?id=sOXU-PEJSgQ>) - [[PyTorch]](<https://github.com/TopoXLab/consistency-ranking-loss>)

### Natural Language Processing

Awesome LLM Uncertainty, Reliability, & Robustness [[GitHub]](<https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness>)


**Conference**

- R-U-SURE? Uncertainty-Aware Code Suggestions By Maximizing Utility Across Random User Intents [[ICML2023]](<https://arxiv.org/pdf/2303.00732.pdf>) - [[GitHub]](https://github.com/google-research/r_u_sure)
- Strength in Numbers: Estimating Confidence of Large Language Models by Prompt Agreement [[TrustNLP2023]](<https://aclanthology.org/2023.trustnlp-1.28/>) - [[GitHub]](https://github.com/JHU-CLSP/Confidence-Estimation-TrustNLP2023)
- Disentangling Uncertainty in Machine Translation Evaluation [[EMNLP2022]](<https://arxiv.org/abs/2204.06546>) - [[PyTorch]](<https://github.com/deep-spin/uncertainties_mt_eval>)
- Investigating Ensemble Methods for Model Robustness Improvement of Text Classifiers [[EMNLP2022 Findings]](<https://arxiv.org/abs/2210.16298>)
- DATE: Detecting Anomalies in Text via Self-Supervision of Transformers [[NAACL2021]](<https://arxiv.org/abs/2104.05591>)
- Calibrating Structured Output Predictors for Natural Language Processing [[ACL2020]](<https://aclanthology.org/2020.acl-main.188/>)
- Calibrated Language Model Fine-Tuning for In- and Out-of-Distribution Data [[EMNLP2020]](<https://aclanthology.org/2020.emnlp-main.102/>) - [[PyTorch]](https://github.com/Lingkai-Kong/Calibrated-BERT-Fine-Tuning)

**Journal**
- How Can We Know When Language Models Know? On the Calibration of Language Models for Question Answering [[TACL2021]](https://arxiv.org/abs/2012.00955) - [[PyTorch]](https://github.com/jzbjyb/lm-calibration)

**Arxiv**

- Gaussian Stochastic Weight Averaging for Bayesian Low-Rank Adaptation of Large Language Models [[arXiv2024]](<https://arxiv.org/pdf/2405.03425>)
- To Believe or Not to Believe Your LLM [[arXiv2024]](<https://arxiv.org/abs/2406.02543>)
- Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling [[arXiv2023]](<https://arxiv.org/abs/2311.08718>)

### Others

**Conference**

- PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness [[CVPR2024]](<https://arxiv.org/pdf/2312.02158.pdf>) - [[Website]](<https://astra-vision.github.io/PaSCo/>)
- Uncertainty Quantification via Stable Distribution Propagation [[ICLR2024]](<https://arxiv.org/abs/2402.08324>)
- Assessing Uncertainty in Similarity Scoring: Performance & Fairness in Face Recognition [[ICLR2024]](<https://arxiv.org/abs/2211.07245>)

**Arxiv**

- Shaving Weights with Occam's Razor: Bayesian Sparsification for Neural Networks Using the Marginal Likelihood - [[arxiv2024]](<https://arxiv.org/pdf/2402.15978>)
- Urban 3D Panoptic Scene Completion with Uncertainty Awareness [[arXiv2023]](<https://astra-vision.github.io/PaSCo/>) - [[PyTorch]](<https://github.com/astra-vision/PaSCo>)

# Datasets and Benchmarks

- SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation [[CVPR2022]](<https://openaccess.thecvf.com/content/CVPR2022/html/Sun_SHIFT_A_Synthetic_Driving_Dataset_for_Continuous_Multi-Task_Domain_Adaptation_CVPR_2022_paper.html>)
- MUAD: Multiple Uncertainties for Autonomous Driving, a benchmark for multiple uncertainty types and tasks [[BMVC2022]](<https://arxiv.org/abs/2203.01437>) - [[PyTorch]](<https://github.com/ENSTA-U2IS-AI/MUAD-Dataset>)
- ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding [[ICCV2021]](<https://arxiv.org/abs/2104.13395>)
- The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection [[IJCV2021]](<https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf>)
- SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation [[NeurIPS2021]](<https://arxiv.org/abs/2104.14812>)
- Uncertainty Baselines: Benchmarks for Uncertainty & Robustness in Deep Learning [[arXiv2021]](<https://arxiv.org/abs/2106.04015>) - [[TensorFlow]](<https://github.com/google/uncertainty-baselines>)
- Curriculum Model Adaptation with Synthetic and Real Data for Semantic Foggy Scene Understanding [[IJCV2020]](<https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/>)
- Benchmarking the Robustness of Semantic Segmentation Models [[CVPR2020]](<https://arxiv.org/abs/1908.05005>)
- Fishyscapes: A Benchmark for Safe Semantic Segmentation in Autonomous Driving [[ICCV Workshop2019]](<https://openaccess.thecvf.com/content_ICCVW_2019/html/ADW/Blum_Fishyscapes_A_Benchmark_for_Safe_Semantic_Segmentation_in_Autonomous_Driving_ICCVW_2019_paper.html>)
- Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming [[NeurIPS Workshop2019]](<https://arxiv.org/abs/1907.07484>) - [[GitHub]](<https://github.com/bethgelab/robust-detection-benchmark>)
- Semantic Foggy Scene Understanding with Synthetic Data [[IJCV2018]](<https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/>)
- Lost and Found: Detecting Small Road Hazards for Self-Driving Vehicles [[IROS2016]](<https://arxiv.org/abs/1609.04653>)

# Libraries

## Python

- Uncertainty Calibration Library [[GitHub]](<https://github.com/p-lambda/verified_calibration>)
- MAPIE: Model Agnostic Prediction Interval Estimator [[Sklearn]](https://github.com/scikit-learn-contrib/MAPIE)
- Uncertainty Toolbox [[GitHub]](<https://uncertainty-toolbox.github.io/>)
- OpenOOD: Benchmarking Generalized OOD Detection [[GitHub]](<https://github.com/jingkang50/openood>)
- Darts: Forecasting and anomaly detection on time series [[GitHub]](<https://github.com/unit8co/darts>)
- Mixture Density Networks (MDN) for distribution and uncertainty estimation [[GitHub]](<https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation>)

## PyTorch

- TorchUncertainty [[GitHub]](<https://github.com/ENSTA-U2IS-AI/torch-uncertainty>)
- Bayesian Torch [[GitHub]](<https://github.com/IntelLabs/bayesian-torch>)
- Blitz: A Bayesian Neural Network library for PyTorch [[GitHub]](<https://github.com/piEsposito/blitz-bayesian-deep-learning>)

## JAX

- Fortuna [[GitHub - JAX]](<https://github.com/awslabs/fortuna>)

## TensorFlow

- TensorFlow Probability [[Website]](<https://www.tensorflow.org/probability>)

# Lectures and tutorials

- Dan Hendrycks: Intro to ML Safety course [[Website]](<https://course.mlsafety.org/>)
- Uncertainty and Robustness in Deep Learning Workshop in ICML (2020, 2021) [[SlidesLive]](<https://slideslive.com/icml-2020/icml-workshop-on-uncertainty-and-robustness-in-deep-learning-udl>)
- Yarin Gal: Bayesian Deep Learning 101 [[Website]](<http://www.cs.ox.ac.uk/people/yarin.gal/website/bdl101/>)
- MIT 6.S191: Evidential Deep Learning and Uncertainty (2021) [[Youtube]](<https://www.youtube.com/watch?v=toTcf7tZK8c>)
- Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users [[IEEE Computational Intelligence Magazine]](https://arxiv.org/pdf/2007.06823.pdf)

# Books

- The "Probabilistic Machine-Learning" book series by Kevin Murphy [[Book]](<https://probml.github.io/pml-book/>)

# Other Resources

Uncertainty Quantification in Deep Learning [[GitHub]](<https://github.com/ahmedmalaa/deep-learning-uncertainty>)

Awesome Out-of-distribution Detection [[GitHub]](<https://github.com/continuousml/Awesome-Out-Of-Distribution-Detection>)

Anomaly Detection Learning Resources [[GitHub]](<https://github.com/yzhao062/anomaly-detection-resources>)

Awesome Conformal Prediction [[GitHub]](<https://github.com/valeman/awesome-conformal-prediction>)

Awesome LLM Uncertainty, Reliability, & Robustness [[GitHub]](<https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness>)

UQSay - Seminars on Uncertainty Quantification (UQ), Design and Analysis of Computer Experiments (DACE) and related topics @ Paris Saclay [[Website]](<https://www.uqsay.org/p/welcome.html/>)

ProbAI summer school [[Website]](<https://probabilistic.ai/>)

Gaussian process summer school [[Website]](<https://gpss.cc/>)
