### A general framework of Bag-of-Visual-Words(BoVW) for action classification

#### REQUIREMENTS

* python-2.7.6 
* libsvm-3.1.8
* numpy-MKL-1.8.1
* scipy-0.14.0

#### Dataset:
* [KTH Action Database](http://www.nada.kth.se/cvap/actions/)
* We follow the normal settings in [previous works](http://www.nada.kth.se/cvap/actions/00sequences.txt) 

#### DESCRIPTION
* Feature extraction: [STIP](http://www.di.ens.fr/~laptev/download.html#stip)
* Feature Encoding: Vector Quantization, Hard assignment
* Pooling and normalization: sum pooling, L1-norm
* Classification: RBF-Kernel SVM


#### CONTACT INFORMATION
* Website: http://csusap.csu.edu.au/~bliang03/
* E-mail: bliang@csu.edu.au