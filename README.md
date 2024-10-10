# MVP-PCLIP (Zero-Shot Point Cloud Anomaly Detection)


> [**IEEE TSMC Under Review**] [**Towards Zero-shot Point Cloud Anomaly Detection: A Multi-View Projection Framework**](https://export.arxiv.org/abs/2409.13162).
>
> by [Yuqi Cheng*](https://hustcyq.github.io/), [Yunkang Cao*](https://caoyunkang.github.io/), [Guoyang Xie](https://guoyang-xie.github.io/), Zhichao Lu, [Weiming Shen](https://scholar.google.com/citations?user=FuSHsx4AAAAJ&hl=en),

## Introduction 
Detecting anomalies within point clouds is crucial for various industrial applications, but traditional unsupervised methods face challenges due to data acquisition costs, early-stage production constraints, and limited generalization across product categories. To overcome these challenges, we introduce the Multi-View Projection (MVP) framework, leveraging pre-trained Vision-Language Models (VLMs) to detect anomalies. Specifically, MVP projects point cloud data into multi-view depth images, thereby translating point cloud anomaly detection into image anomaly detection. Following zero-shot image anomaly detection methods, pre-trained VLMs are utilized to detect anomalies on these depth images. Given that pre-trained VLMs are not inherently tailored for zero-shot point cloud anomaly detection and may lack specificity, we propose the integration of learnable visual and adaptive text prompting techniques to fine-tune these VLMs, thereby enhancing their detection performance. Extensive experiments on the MVTec 3D-AD and Real3D-AD demonstrate our proposed MVP framework's superior zero-shot anomaly detection performance and the prompting techniques' effectiveness. Real-world evaluations on automotive plastic part inspection further showcase that the proposed method can also be generalized to practical unseen scenarios.

## Overview of MVP-PCLIP
<img src="./Imgs/F5.png" width="800px">

## 🛠️ Getting Started

### Installation



### Dataset Preparation 
Please download our processed visual anomaly detection datasets to your `DATA_ROOT` as needed. 

| Dataset | Google Drive | Baidu Drive | Note
|------------|------------------|------------------| ------------------|
| MVTec 3D-AD    | [Google Drive]() | [Baidu Drive]() | Original |
| Real3D-AD    | [Google Drive]() | [Baidu Drive]() | Original |
| MVTec3D-2D    | [Google Drive]() | [Baidu Drive]() | Original |
| Real3D-2D    | [Google Drive]() | [Baidu Drive]() | Original |





### preprocess




### Train


### Test


## Main Results

### 1. Point-wise on MVTec 3D
<img src="./Imgs/T1.png" width="800px">

### 2. Point-wise on Real3D
<img src="./Imgs/T2.png" width="800px">

### 3. Object-wise both on MVTec 3D and Real3D
<img src="./Imgs/T3.png" width="400px">

## 💘 Acknowledgements
Our work is largely inspired by the following projects. Thanks for their admiring contribution.

- [VAND-APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN)
- [CPMF](https://github.com/caoyunkang/CPMF)





## Citation

If you find this project helpful for your research, please consider citing the following BibTeX entry.

```BibTex

@inproceedings{AdaCLIP,
  title={Towards Zero-shot Point Cloud Anomaly Detection: A Multi-View Projection Framework},
  author={},
  booktitle={},
  year={2024}
}

```




