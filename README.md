# Introduction
This is the source code of our TCSVT 2019 paper "Zero-shot Cross-media Embedding Learning with Dual Adversarial Distribution Network", Please cite the following paper if you find our code useful.

Jingze Chi and Yuxin Peng, "Zero-shot Cross-media Embedding Learning with Dual Adversarial Distribution Network", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Feb. 2019. [[PDF]](http://39.108.48.32/mipl/download_paper.php?fileId=20193)

# Preparation 
Our code is based on tensorflow 1.4, and tested on Ubuntu 16.04 LTS, python 2.7.

# Usage
Data Preparation: We use [PKU XMediaNet dataset](http://39.108.48.32/mipl/XMediaNet/) as example, and the data should be put in ./data/. The data files can be download from the [link](http://39.108.48.32/mipl/tiki-download_file.php?fileId=1012) and unzipped to the above path.

Run DADN.py to train models and calculate mAP.

# Our Related Work
If you are interested in cross-media retrieval, you can check our recently published overview paper on IEEE TCSVT:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol.28, No.9, pp.2372-2385, 2018. [[PDF]](http://39.108.48.32/mipl/download_paper.php?fileId=201823)

Welcome to our [Benchmark Website](http://39.108.48.32/mipl/XMediaNet/) and [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
