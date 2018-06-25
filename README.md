# Neural Natural Language Inference Models Enhanced with External Knowledge
Source code for "Neural Natural Language Inference Models Enhanced with External Knowledge" based on Theano.
If you use this code as part of any published research, please acknowledge the following paper.

**"Neural Natural Language Inference Models Enhanced with External Knowledge"**
Qian Chen, Xiaodan Zhu, Zhen-Hua Ling, Diana Inkpen and Si Wei. _ACL (2018)_ 

```
@InProceedings{Chen-Qian:2018:ACL,
  author    = {Chen, Qian and Zhu, Xiaodan and Ling, Zhen-Hua and Inkpen, Diana and Wei, Si},
  title     = {Neural Natural Language Inference Models Enhanced with External Knowledge},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018)},
  month     = {July},
  year      = {2018},
  address   = {Melbourne, Australia},
  publisher = {ACL}
}
```
Homepage of the Qian Chen, http://home.ustc.edu.cn/~cq1231/

## Dependencies
To run it perfectly, you will need (recommend using Ananconda to set up environment):
* Python 2.7.13
* Theano 0.9.0

## Running the Script
1. Download and preprocess 
```
cd data
bash fetch_and_preprocess.sh
```

2. Train and test model
```
cd scripts/kim/
bash train.sh
```
The result is in `scripts/kim/log.txt` file.
3. Analysis the result for dev/test set (optional)
```
bash test.sh
```
