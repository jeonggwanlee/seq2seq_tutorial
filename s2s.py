

vab_size=256 # # of ASCII Code
target_vocab_size=vocab_size # the model selects (classify) one of 256 classes (ASCII codes) per time unit
learning_rate=0.1
buckets=[(12, 12)] # Because seq2seq does batch learning, it buckets input by length. This time we will only deal with one bucket.
PAD=[0] # If the input / target sentence is smaller than the bucket size, pad 0.
GO=[1] # Decoder RNN puts the symbol GO as the first input.
batch_size=1
input_string = "Hello World" 
target_string = "How are you"

input_PAD_size = buckets[0][0] - len(input_string) # Decide how much PAD you want to input/
target_PAD_size = buckets[0][0] - len(target_string) - 1 # Decide how much PAD you want to target.
input_data = (map(ord, input_string) + PAD * input_PAD_size) * batch_size # Change the input text to a list of ASCII codes.
target_data = (GO + map(ord, target_string) + PAD * target_PAD_size) * batch_size # Change target phrase to list of ASCII codes.
target_weights= ([1.0]*12 + [0.0]*0) * batch_size  # The number of actual valid (loss counted) number of characters 
                                                    # excluding PAD in the target sentence.

## Set up the architecture
class Seq2Seq(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
        self.
