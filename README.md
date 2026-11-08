# Rock-Paper-Scissors: custom CNN with a live webcam GUI

## Project files

The training work is in the Colab notebook `rps_training_final.ipynb`, which covers the split, hand crop, augmentation, training, evaluation, cross-validation and model export. The webcam application is `rps_gui.py` and runs on a normal computer. The trained models used by the GUI are `crop_cnn4.h5` and `aug_cnn4.h5`, which hold the weights together with the preprocessing settings, and the same models also exist as PyTorch `.pth` checkpoints. The file `split_by_class.csv` records which set every image belongs to, and the images themselves live in a `Dataset` folder with one sub-folder each for Paper, Rock and Scissors, which is the standard ImageFolder layout.

## Dataset

All images were photographed by hand with two phones, a Redmi 12C and a Google Pixel, against varied backgrounds and in varied lighting. The dataset holds 728 Paper images, 599 Rock images and 696 Scissors images, which makes 2,023 in total. Every class has more than 500 images and the total is above 2,000. The classes are deliberately unbalanced, and this is handled with class weights in the loss function.

## Method

The order of the processing steps matters, and it is the same for every model. First the list of files is split into 70% training, 15% validation and 15% test images, chosen at random with the same share of each class in every part and a fixed seed of 42. The split happens before anything is cropped, resized or augmented, and the test set contains 304 images, made of 109 Paper, 90 Rock and 105 Scissors.

Next comes a hand crop that uses classical image processing and no trained detector. It finds skin-coloured pixels in the YCrCb colour space, cleans the mask with morphological operations, takes the largest skin region and cuts a square around it with a 25% margin, so some background stays in the picture. If no hand is found, the centre of the picture is used instead. The result is a 256 by 256 working copy of each image.

Augmentation is applied to the training images only and runs on the GPU, with a new random draw every epoch. It consists of rotation over the full circle, zoom, horizontal mirroring, changes in contrast, saturation and colour tint, a degradation of picture quality through lower resolution, blur and noise, and a small randomly erased patch. After augmentation the image is resized to 128 by 128 pixels, which is the last geometric step, and then normalised with the per-channel mean and standard deviation of the training images. Validation and test images are cropped, resized and normalised in the same way but are never augmented.

## Models

All models are written with plain PyTorch layers. The linear model flattens the image and applies a single linear layer, and it serves as the baseline. The CNN is built from blocks of a 3 by 3 convolution, batch normalisation, ReLU and max pooling, with three, four or five blocks, followed by global average pooling and a small classifier head with dropout. The two-branch model runs two CNN branches side by side on the same image, one with 3 by 3 filters and one with 7 by 7 filters, and joins their features before the classifier.

Training uses the Adam optimiser with a learning rate of 0.001 that is halved when the validation loss stops improving, a batch size of 32 and a class-weighted cross-entropy loss. A run lasts at most 80 epochs and stops early when the validation macro-F1 has not improved for 12 epochs. The best epoch is kept, and the test set is evaluated only once at the end.

## Logging

Training and validation loss and accuracy, validation macro-F1, the learning rate, and the final test metrics and confusion matrix are logged to Weights & Biases in the project `rps-acv-assignment3`.

## Results

The first complete run used a random class-balanced split and the 304 test images, with an earlier augmentation that still included shifts, brightness changes and rotation of up to 30 degrees. The four-block CNN without augmentation reached 91.8% test accuracy with a macro-F1 of 0.920. The same CNN with augmentation reached 87.2% with a macro-F1 of 0.876, and the two-branch CNN with four blocks and augmentation reached 84.2% with a macro-F1 of 0.847. The three-block CNN with augmentation reached only 56.9%, and the linear model 40.8%, so both of those trained poorly under heavy augmentation. Both four-block CNNs are above the 85% requirement of the assignment. The most common mistake in every model is Paper being predicted as Scissors.

These numbers should be read with care. The photos were taken in bursts, so near-identical pictures can fall on both sides of the split, and the test scores are probably a little optimistic. On these phone photos the model without augmentation scored higher, but augmentation is meant to help on a different camera, so the live webcam test matters more than the test-set score.

The live accuracy on the webcam is measured with the accuracy test in the GUI, which records 100 frames per gesture and scores every loaded model on exactly the same frames. [Add the webcam results for each model here and name the final model used for the demonstration.]

## Running the project

To train, put the dataset folder in Google Drive, open `rps_training_final.ipynb` in Colab and set the runtime to a T4 GPU. Set the `PROJECT` variable in the settings cell to the dataset folder, run the cells in order and paste your Weights & Biases API key when it is requested. The trained models are saved as `.pth` and `.h5` files in a `checkpoints_aug` folder next to the dataset, and the `.h5` files can be downloaded for the GUI.

To run the GUI on your own computer, install the packages torch, numpy, pillow, opencv-python and h5py with pip, put the `.h5` files named in `MODEL_FILES` at the top of `rps_gui.py` in the same folder as the script, and start it with `python rps_gui.py`. Hold your hand inside the green box on the camera preview. A blue square shows the crop that the model actually receives, and it turns red when no hand was found. The prediction and the probability of each class are shown live, and the text "No hand" appears when the box contains too little skin colour. For the accuracy test, choose the gesture you are showing and press Start, and after a three second countdown the GUI records the chosen number of frames and scores all loaded models on them. The results can be saved as a CSV file. The GUI prepares every frame exactly as the training code does, by cropping, resizing to 256 by 256 and then to 128 by 128, and normalising with the mean and standard deviation stored inside each `.h5` file.

## Limitations

The training photos come from two phones while the webcam is a different camera, so the webcam accuracy can be lower than the test-set accuracy. The models always answer with one of the three classes, and the decision that no hand is present is made by a skin-colour rule in the GUI rather than by the model, so skin-coloured objects can fool it. The hand crop depends on skin colour and can fail on skin-toned backgrounds or in poor lighting, in which case it falls back to the centre of the box. Near-duplicate burst photos were not grouped when splitting the data, so the test results are likely to be optimistic.

## Possible improvements

Adding webcam frames to the training data is the most promising improvement, since it closes the gap to the real camera. Other options are to train the five-block CNN and the two-branch model with the final augmentation, to run five-fold cross-validation, to add a model with one branch on the colour image and another on a Sobel edge map, and to add a fourth "no hand" class so that the skin-colour rule is no longer needed.
