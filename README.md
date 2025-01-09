# Sparse Enhanced Network (SparseEnNet)

This is a demo of paper Sparse Enhanced Network: An Adversarial Generation Method for Robust Augmentation in Sequential Recommendation

## Requirements

```
Python = 3.9.0
torch >= 1.12.1
faiss-gpu
scipy
tqdm
nni
```

## Evaluate

We provide a checkpoint of our model in Beauty. You can evaluate the model by running the following script:

```
python3 main.py --data_name Beauty --num_hidden_layers 1 --eval_id 1 --do_eval
```

## Training

Run following script to train SparseEnNet:

```
python3 main.py --data_name Beauty --gpu_id 0 --batch_size 256
```

## LICENSE

This project uses part of code from project [ICLRec](https://github.com/YChen1993/ICLRec) under following license 

```
BSD 3-Clause License

Copyright (c) 2021, Salesforce.com, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of Salesforce.com nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

