# CIL - Road Segmentation project
Segmenting an image consists in partitioning an image into multiple segments (formally 
one has to assign a class label to each pixel).
For this problem,  144 aerial images are provided which were acquired from Google Maps. For each image
ground truth labels are given where each pixel gets assigned a probability in [0,1] that it 
is {road=1, background=0}. The goal is to train a classifier to segment roads in these images, 
i.e. assign a probabilistic label {road=1, background=0} to each pixel.

## Required datasets
### Kaggle competition dataset
The training requires the dataset from the kaggle competition. https://www.kaggle.com/competitions/48353/leaderboard/download/public

### Masked/Inpainted
Both datasets can be downloaded here: https://polybox.ethz.ch/index.php/s/7ndkvkoVd68JEoh
They contain the described datasets in our paper and are required to run the experiments. The masked images were created with the script images/mask_images.py. Inpainting was done with RePaint (https://github.com/andreas128/RePaint) using their code in test.py with the test_p256_thin.yml configuration file. Finally, the images were reassembled to their full resolution using the script inpainting/reassemble_images.py.

### DataGlobe
1. Make sure you download the full dataset from https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset/code
2. Unzip it and rename the folder to DeepGlobe_Road_Extraction. 
3. Create aa folder named "DeepGlobeSampled" and inside the folders "images" and "groundtruth" where the images are copied into.
4. Run the Notebook deepGlobe_sample.ipynb, which preprocesses the images of the DataGlobe dataset to the desired format

## Reproducing experiments
- To perform the k-fold, all experiments for each method described in our paper can be run with the Experiment-kfold.ipynb Notebook.
- To perform the experiment for the DeepGlobe Dataset use the Notebook Experiment-kfold_deepGlobe.ipynb
- To run experiments with the U-Net with attention, make sure to inclue attn_unet.py to your Workspace and instance it with UTnet().
- To run each method, create the augmentation folder (with the Experiment-kfold.ipynb Notebook) and create the ImageDatasets with the
desired parameters accordingly (e.g. use_augmentation=True). In case you want to switch between inpainted or masked images, just replace
the folder path accordingly in the ImageDataset class and in the creation of the augmentation folder.
- Run all cells in the Notebook to create a submission in .csv format

## Final submission
Our final submission was also run with Experiment-kfold.ipynb and using masked images, but by combining the predicitions of all folds. The code to generate the the submission with the averages is in the last cell in this Notebook. To reproduce the final submission, make sure to run the Notebook and set variable inpainted_path on "masked", so it takes the images from the masked folder.
