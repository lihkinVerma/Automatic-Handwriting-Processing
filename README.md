# Automatic-Handwriting-Processing
Intelligent Handwriting Character Recognition is one of the field left very less touched and worked upon. Various OCR engines work for English language but that too not handwritten. The Aim to work on this project is to the processing of Handwritten images and extract something menaingful out of it. The stepwise approch used is quiet clear from the number system  used for various directories.

1. line-word segmentation from any given input image 
2. character segmentation from any gven input image
3. ML dataset of EMNIST conversion from format available into csv files
4. Building CNN model using EMNIST dataset
5. testing the model on test dataset to find accuracy
6. Comparator of segmented characters to find similar images
7. OCR and Audio analysis to convert any input text image and produce audio of same

The output of various working modeules is as follows:
## line-word segmentation from any given input image 
![1 line-word-seg](https://user-images.githubusercontent.com/32717195/43880195-08ea23bc-9bc5-11e8-881e-9cd5053f262f.JPG)

## contour formation
![2 contour](https://user-images.githubusercontent.com/32717195/43880196-0a44ceba-9bc5-11e8-88c1-92863db10482.JPG)

## count of lines and words representation
![3 l-w-seg-res](https://user-images.githubusercontent.com/32717195/43880198-0baccd70-9bc5-11e8-9290-6097b4ea3680.JPG)

## character segmentation from any gven input image
![4 charseg](https://user-images.githubusercontent.com/32717195/43880200-0d027530-9bc5-11e8-969d-6e6a87088702.JPG)

## Encoding scheme used
![5 encoding](https://user-images.githubusercontent.com/32717195/43880204-10bf4536-9bc5-11e8-8333-23d7ba2c1047.JPG)

## Training of Convolutional neural network
# basic workflow
![cnn](https://user-images.githubusercontent.com/32717195/43880221-1ff1d2bc-9bc5-11e8-8b17-f56ef0f8f8f5.jpg)
# Real implementation
![6 cnn training](https://user-images.githubusercontent.com/32717195/43880206-12446012-9bc5-11e8-8893-327a23128dc9.JPG)

## Training epochs and loss calculation
![7 epochs](https://user-images.githubusercontent.com/32717195/43880209-151cd92c-9bc5-11e8-8f98-bfa2ee678ac5.JPG)

## RMSE value calculations
![8 rmse](https://user-images.githubusercontent.com/32717195/43880212-181f3a0c-9bc5-11e8-9bfc-041723c8aff9.jpg)

## OCR and Text to speech Analysis
![9 ocr-working](https://user-images.githubusercontent.com/32717195/43880218-1cd6b318-9bc5-11e8-8e1a-a3f7f1c5c5e0.JPG)
