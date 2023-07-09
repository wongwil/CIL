# CIL - Road Segmentation project
Segmenting an image consists in partitioning an image into multiple segments (formally 
one has to assign a class label to each pixel).
For this problem,  144 aerial images are provided which were acquired from Google Maps. For each image
ground truth labels are given where each pixel gets assigned a probability in [0,1] that it 
is {road=1, background=0}. The goal is to train a classifier to segment roads in these images, 
i.e. assign a probabilistic label {road=1, background=0} to each pixel.
