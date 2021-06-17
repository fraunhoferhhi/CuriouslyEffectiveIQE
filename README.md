This repository provides the reference implementation for the paper\
**Curiously Effective Features for Image Quality Prediction**\
which has been accepted for publication at ICIP 2021 ([Link to preprint](https://arxiv.org/abs/2106.05946)). 

```
@misc{becker2021curiously,
      title={Curiously Effective Features for Image Quality Prediction}, 
      author={S\"oren Becker and Thomas Wiegand and Sebastian Bosse},
      year={2021},
      eprint={2106.05946},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

You can reproduce our results with the following steps:

1. Set up a python environment using the provided *CuriousFeatures.yml* file.
2. Download the datasets and use *format_datasets.py* to bring the data into the expected format.
3. Run *bash feature_extraction.sh* in your terminal. This will extract features according to the paper from all images in all databases. We provide two scripts here, *feature_extraction.sh* and *feature_extraction_reproduce.sh*. *feature_extraction.sh* uses PyTorch and runs considerably faster than *feature_extraction_reproduce.sh*, however, results will likely deviate slightly from the results reported in the paper. If you want to reproduce our results and do not care about computational speed, you can use *feature_extraction_reproduce.sh*.
4. Open the jupyter notebook *Experiments.ipynb* and follow the instructions therein to reproduce reported correlations.

### License

The copyright in this software is being made available under this Software
Copyright License. This software may be subject to other third party and
contributor rights, including patent rights, and no such rights are
granted under this license.
Copyright (c) 1995 - 2021 Fraunhofer-Gesellschaft zur Förderung der
angewandten Forschung e.V. (Fraunhofer)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted for purpose of testing the functionalities of
this software provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
