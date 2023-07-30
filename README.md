# CIL - Road Segmentation project
Segmenting an image consists in partitioning an image into multiple segments (formally 
one has to assign a class label to each pixel).
For this problem,  144 aerial images are provided which were acquired from Google Maps. For each image
ground truth labels are given where each pixel gets assigned a probability in [0,1] that it 
is {road=1, background=0}. The goal is to train a classifier to segment roads in these images, 
i.e. assign a probabilistic label {road=1, background=0} to each pixel.

## Generating datasets
### Inpainted
TODO

### DataGlobe
1. Make sure you you download the full dataset from https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset/code
2. Unzip it and rename the folder to DeepGlobe_Road_Extraction. 
3. Creat a folder named "DeepGlobeSampled" and inside the folders "images" and "grountruth" where the images are copied into.
4. Run the Notebook deepGlobe_sample.ipynb, which preprocesses the images of the DataGlobe dataset to the desired format


## Reproducing experiments
- To perform the k-fold, all experiments for each method can be run with the Experiment-kfold.ipynb Notebook.
- To perform the experiment for the DeepGlobe Dataset use the Notebook Experiment-kfold_deepGlobe.ipynb
- To run the methods described in our paper, create the augmentation folder with the Notebook and create the ImageDatasets with the
desired parameters accordingly (e.g. use_augmentation=True)
