# XGrad-CAM implementation in Pytorch 

Code for the paper:
### [Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs](https://arxiv.org/pdf/2008.02312v4)

to be presented at **BMVC 2020**,
<br>
Authors:
<br>
Ruigang Fu,
Qingyong Hu,
Xiaohu Dong,
Yulan Guo,
Yinghui Gao and
Biao Li,
<br>

----------

### XGrad-cam.py
XGrad-CAM is a CNN visualization method, try to explain why classification CNNs predict what they predict. It is class-discriminative, efficient and able to highlight the regions belonging to the objects of interest.

<img src="https://github.com/Fu0511/XGrad-CAM/tree/main/examples/XGrad-CAM.png" width="70%">

The main difference between XGrad-CAM and Grad-CAM locates at line 116 - line120:
#####  XGrad-CAM
`X_weights = np.sum(grads_val[0, :] * target, axis=(1, 2))`

`X_weights = X_weights / (np.sum(target, axis=(1, 2)) + 1e-6)`
#####  Grad-CAM 
`weights = np.mean(grads_val, axis=(2, 3))[0, :]`

Usage: `python XGrad-cam.py --image-path <path_to_image> --target-index <class_of_interest>` for CPU computation, add `--use-cuda` for GPU acceleration.

Example: `python XGrad-CAM.py --image-path ./examples/ILSVRC2012_val_00000077.JPEG --target-index 159 --use-cuda`

Output: `class of interest:  n02087394 Rhodesian ridgeback`

Results:

![Grad-CAM](https://github.com/Fu0511/XGrad-CAM/tree/main/examples/cam.jpg) ![XGrad-CAM](https://github.com/Fu0511/XGrad-CAM/tree/main/examples/X_cam.jpg)

left is Grad-CAM, right is XGrad-CAM

----------

### Proof_verify.py
This is a simple script of experimental proof for our statement that given an arbitrary layer in ReLU-CNNs, there
exists a specific equation between the class score and the feature maps of the layer (Eq.(5) in our paper).

Usage: `python Proof_verify.py --image-path <path_to_image> --target-index <class_of_interest>`

For any class of interest, the result will show that `class_score-gradients*feature-bias_term=0`

----------

These codes are based on https://github.com/jacobgil/pytorch-grad-cam.
Thanks to the author Jacob Gildenblat for the beautiful original code.

If these codes are useful to you, please cite our work:
```
@misc{fu2020axiombased,
    title={Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs},
    author={Ruigang Fu and Qingyong Hu and Xiaohu Dong and Yulan Guo and Yinghui Gao and Biao Li},
    year={2020},
    eprint={2008.02312},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
