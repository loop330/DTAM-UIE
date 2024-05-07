# DTAM-UIE
This repository is the official PyTorch implementation of DTAM-UIE:An Underwater Image Enhancement Method Based on Dual Transformer Aggregation and Multi-scale Feature Fusion.

Underwater suspended particles can induce light attenuation and scattering, leading to color distortion, low contrast, and blurred details in underwater imagery. This phenomenon has implications for the application of advanced visual tasks in such environments. We propose a method for underwater image processing based on dual Transformer aggregation and multi-scale feature fusion(DTAM-UIE) to address the above issues. First, underwater feature images are extracted by combining the channel self-attention Transformer(CFormer) with the Bi-level routing dynamic sparse self-attention Transformer(BiFormer). We have further enhanced the
channel self-attention mechanism by introducing additional a blueprint separable convolutions branche, enabling the Transformer to capture both global and local features effectively. Then, a parallel-channel spatial attention block is constructed to fuse features extracted by the dual Transformers. Additionally, we design a multi-scale fusion block to aggregate different scale features of the decoder part of the network model to enhance the adaptability of the network model to different underwater environments. Finally, we construct an efficient post-processing block for the U-shaped network’s output layer to fuse each part’s output features. This method effectively improves color distortion, enhances contrast, and sharpens image details. Experimental results demonstrate that this the method outperforms other qualitative and quantitative comparison methods and can better recover color and texture details
#
## Training</br> 
We used five benchmark datasets, which you can download at the following link:[UIEB](https://li-chongyi.github.io/proj_benchmark.html);   [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset);   [LSUI](https://drive.google.com/file/d/10gD4s12uJxCHcuFdX9Khkv37zzBwNFbL/view);   [U45](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-/tree/master/upload/U45);   [RUIE](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark).   Put dataset into the folder "data".The structure of the data folder is as follows:</br>
_ _ _**data**</br>
&emsp; _ _ _ dataset_name</br>
&emsp; &emsp; _ _ _ train</br>
&emsp; &emsp; &emsp; _ _ _ iamge</br>
&emsp; &emsp; &emsp; _ _ _ reference</br>
&emsp; &emsp; _ _ _ val</br>
&emsp; &emsp; &emsp; _ _ _ iamge</br>
&emsp; &emsp; &emsp; _ _ _ reference</br></br>
**Environmental requirements:**</br>
- torch == 2.0.0
- torchvision == 0.15.1
- tqdm == 4.61.2
- timm == 0.9.8
- scikit-image == 0.21.0
- scipy == 1.10.1
- opencv-python == 4.8.1</br>

After the data set file is properly placed and the necessary environment is configured, run the train.py file for model training.

## Testing</br> 
We will test our model at the end of each training round, you can also load the model weight file after the overall training to test.   The model weight files are saved in the "weight" folder.   The structure of the result folder is as follows:</br>
_ _ _ **result**</br>
&emsp; _ _ _ dataset_name</br>
&emsp; &emsp; _ _ _ val_output_mid</br>
&emsp; &emsp; _ _ _ val_output</br>

The "val_output" folder stores the optimal output.  The "val_output_mid" folder is used to store the intermediate results of the entire training process.


