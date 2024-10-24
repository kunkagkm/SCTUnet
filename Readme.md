# SCTUnet
Code for paper “A Self-Configuring Transformer Segmentation Method for Welding Radiographic Defect Detection in Steel Pipes” 

Parts of codes are borrowed from [nnUNet](https://github.com/MIC-DKFZ/nnUNet), about [Dataset](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) you can follow the settings in nnUNet for path configurations and preprocessing procedures .

## Training
nnUNetv2_train 5 2d 0

## Val
nnUNetv2_predict -i .\Dataset\nnUNet_raw\Dataset5\imagesTs -o .\Predict\predict5 -d 5 -p nnUNetPlans -c 2d -f 0 --save_probabilities




