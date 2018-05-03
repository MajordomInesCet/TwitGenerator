# TwitGenerator

This is the code used to generate tweets from our beloved spanish all-powered non-populist all-common-sense leaders. Some of this tweets can be found in this twitter account: @MajordomInesCet

The user_data.py script collects all tweets from a single user and stores it in a serialized file. You can use it to generate data from any twitter user. Its just a bunch lf strings tbh.

The cleaning notebook cleans the data a little bit and both LM.py or .ipynb are simple char-rnnlm trained on the data. you can tune some parameters if you want to train one
The rnnlm is basically the keras example with a different layer.

## Thanks
Thanks to all the people that works in open source libraries like:
1. [Keras](https://keras.io)
2. [Numpy](http://www.numpy.org)
3. [Spacy](https://spacy.io)
4. [sci-kit](http://scikit-learn.org/stable/)

Most importantly I want to thank to every single one of us who fought and is fighting for the Catalan Independenc
