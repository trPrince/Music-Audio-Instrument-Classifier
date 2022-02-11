# Music-Audio-Instrument-Classifier
Determines the different instruments played in a mixed music audio

First create the following folders:
  ./clean, ./models, ./logs, ./Random Songs, ./wavfiles
  
Then add your training and testing datasets in ./wavfiles/training and ./wavfiles/testing respectively.
The names of the directories inside your training dataset and testing dataset should match exactly as they will be the classes of our classifier model.

After that run clean.py by opening the terminal and running: python clean.py
Followed by opening the terminal and running: python train.py

Now, predict by opening the terminal and running: python predict.py

Now if you want to predict for a random song. Add that song to the directory ./Random Songs/X Song
Change the clean.py file's source and destination directory path so that it cleans our random song and keeps the cleaned segments in the directory ./Random Songs/Testing/X Song
Add the blank directories matching those of the training dataset present in ./wavfiles/training to the ./Random Songs/Testing directory.

Finally, run python predict_x.py in the terminal to get prediction.

