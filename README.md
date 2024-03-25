# SCTUnet
Code for paper “A self-configuring Transformer Deep Network for welding radiographic image defect segmentation” 

Parts of codes are borrowed from [nnUNet](https://github.com/MIC-DKFZ/nnUNet), about dataset you can follow the settings in nnUNet for path configurations and preprocessing procedures [Dataset](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

## Training
nnUNetv2_train 5 2d 0

## Val
nnUNetv2_predict -i .\Dataset\nnUNet_raw\Dataset5\imagesTs -o .\Predict\predict5 -d 5 -p nnUNetPlans -c 2d -f 0 --save_probabilities




