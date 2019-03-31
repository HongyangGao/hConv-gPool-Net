# hConv-gPool-Net
TensorFlow implementation of Learning Graph Pooling and Hybrid Convolutional Operations for Text Representations (WWW19)

Created by [Hongyang Gao](http://people.tamu.edu/~hongyang.gao/) at Texas A&M University,
[Yongjun Chen](https://www.eecs.wsu.edu/~ychen3/) at Washington State University, and
[Shuiwang Ji](http://people.tamu.edu/~sji/) at Texas A&M University.

## Introduction

We propose novel graph pooling layer and hybrid convolutional layer for text representation learning. It has been accepted in WWW19.

Detailed information about hConv-gPool-Net is provided in https://arxiv.org/abs/1901.06965.

## Citation

```
@article{gao2019learning,
  title={Learning Graph Pooling and Hybrid Convolutional Operations for Text Representations},
  author={Gao, Hongyang and Chen, Yongjun and Ji, Shuiwang},
  journal={arXiv preprint arXiv:1901.06965},
  year={2019}
}
```

## Results

| Models          | AG's News | DBPedia | Yelp Polarity
|-----------------|-----------|---------|--------------|
| Word-level CNN  | 8.55\%    | 1.37\%  | 4.60\% 
| Char-level CNN  | 9.51\%    | 1.55\%  | 4.88\% 
| GCN-Net         | 8.64\%    | 1.69\%  | 7.74\% 
| GCN-gPool-Net   | 8.09\%    | 1.44\%  | 5.82\% 
| hConv-Net       | 7.49\%    | 1.02\%  | 4.45\% 
| hConv-gPool-Net | 7.09\%    | 0.92\%  | 4.37\% 

## Configure the network

All network hyperparameters are configured in main.py.
