# Overview

The purpose of this task is to test your knowledge and capabilities using neural networks and basic computer vision to solve problems. 

# The Task

Oh no! Our facial recognition data-set got all jumbled up. We have thousands of images, but some of them have been 
rotated left, rotated right, and turned upside-down! 

The upright images are like this:

![](https://imgur.com/WY6x3RG)

And some have been jumbled up like this:

![](https://imgur.com/mFUm3E8)
![](https://imgur.com/73Obelm)
![](https://imgur.com/JWdJl0B)

The task is to create a neural network which takes an image of a face as input and returns its orientation: `upright`, 
`rotated_left`, `rotated_right`, or `upside_down`.

## Data Format

### Inputs

The input is a folder full of images.

### Ground Truth

The ground truth is a CSV with the image file-name and the label for it, for example:

```
fn,label
0-10049200_1891-09-16_1958.jpg,rotated_left
0-10110600_1985-09-17_2012.jpg,rotated_left
0-10126400_1964-07-07_2010.jpg,rotated_left
0-1013900_1917-10-15_1960.jpg,upside_down
0-10166400_1960-03-12_2008.jpg,upright
0-102100_1970-10-09_2008.jpg,rotated_left
0-1024100_1982-06-07_2011.jpg,rotated_right
```

## Training Data

The training data is a set of images and the ground truth for them.

[**Download Link**]()

## Test Data

The test data is a set of images without the ground truth.

[**Download Link**]()

## Evaluation

You will be evaluated based on how many images were correctly classified. **You must submit a test.preds.csv which we can
use the attached `eval.py` to run with:**

    python eval.py test.truth.csv test.preds.csv

To give you an estimate, we got a model with 98% correct on the training set and 89% correct on our test set, without tweaking 
it too much - your results should be around this range or better.

# Submission

When you are done, submit the following:

* Your prediction csv file on the test data in the correct format (see above)
* All the code needed to train and run your network to produce that prediction from
  scratch, along with instructions on how to run the code
* A short description of the approach you took and how you arrived at the solution 
  you did

Good luck!
