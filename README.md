

# These programs are modification to existing TACOTRON 2 from NVIDIA, purpose of these modification was to feed in phonemes and get EMA trajectory from the model, without giving in any time related information

Program initially is being trained with one test subject, ANKUR. We've 460 sentences from MOCHA TIMIT from USC. So we feed in phoneme, with maximum padded length, which is 60. And we expect output EMA which are of maximum length 466. 

<!-- ### Following feature engineering was done on dataset:
  1. Make a set of all the characters of Czech and [Indian Names](https://gist.github.com/mbejda/7f86ca901fe41bc14a63), for doing one hot encoding. 
  2. Break each name in a list, and assign each character to a one hot encoded vector.
  3. For label we're assigning a list of [1, 0] for male and [0, 1] for female. This will be used as we're using *categorical crossentropy* method, as our criterion.  -->
<!--  
### Structure of Model used:
I'm using *Bidirectional LSTM of around 524 units with 2 layers*, as it makes sense an algorithm if feeded from end of a name, will learn quite well, and increase in accuracy was observed. So no reason why it shouldn't be used
I'm also using callbacks in Keras model, Early_Stopping with patience 10, and ReduceLROnPlateau, truth be told, model isn't that complicated we didn't need to use this, but no harm in using them


### Accuracy
We're getting accuracy of around **80 to 85%**, not bad since we're not explicitly training on Czech names, still we get that, that's pretty good.


### How to run this program

>> python3 Martin_keras.py "Martin" "Martini"

Basically add any number of names, adding space in between them.


Also you need to extract weights inside tmp, they're in ZIP for compression. Extract them and put them in /tmp -->
