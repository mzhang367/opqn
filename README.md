# Orthonormal Product Quantization Network for Scalable Face Image Retrieval
Official Pytorch implementation of "[Orthonormal Product Quantization Network for Scalable Face Image Retrieval](https://arxiv.org/abs/2107.00327) (OPQN)"

# Overall training procedure of OPQN
<img src="/figures/overview.png" alt="drawing" width="75%"/>
<p></p>

# Usage
## Training
Suppose that the FaceScrub dataset is downloaded and unzipped under the directory `./data/facescrub`. Follow our data configuration in the paper, run the below command to train a 36-bit deep product quantizaion model using 6 codebooks with 64 codewords per codebook, and default hyperparameters.
 ```
 python opqn_train.py --dataset facescrub --save your_model_name.tar --len 36 --num 6 --words 64
 ```
 ```python opqn_train.py --help```will provide detailed explanation of each argument.
