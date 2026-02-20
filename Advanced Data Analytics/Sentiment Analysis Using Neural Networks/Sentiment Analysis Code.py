import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
# makes tensorflow depreciation warnings quiet [In-Text Citation: (user1315789 & Freeman, 2020)]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# there are deprecated functions in keras that create warning messages in tensorflow, 
# this disables those warnings [In-Text Citation: (Ghilas BELHADJ & serv-inc, 2016)]
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import TextVectorization, Dropout, Embedding, GlobalAveragePooling1D, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from keras_tuner import RandomSearch

nltk.download('stopwords')
nltk.download('wordnet')
class text:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# region Functions
# Hyper parameter tuning function [In-Text Citation: (Dutta, 2024)]
def build_model(hp):
   """
   build model: runs the designated amount of tests to find the most accurate neural network
   """
   model = Sequential()
   model.add(Embedding(input_dim=vocab_size, output_dim=dimensions)) # embedding layer
   model.add(GlobalAveragePooling1D())
   model.add(Dropout(rate=0.2, seed=101))
   # picks a random amount of layers to add
   for i in range(hp.Int('n_layers', 1, 5)):
      model.add(Dense(hp.Int(f'dense_{i}',min_value=32,max_value=512,step=32), activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   model.compile(optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
               loss='binary_crossentropy',
               metrics=['accuracy'])
   return model
# endregion
df = pd.read_csv('imdb_labelled.txt', sep='  \t', header=None, names=['Review', 'Sentiment'], engine='python')

print (text.BOLD + "\n-- Begin Data Analytics --\n" + text.END)

# region Analysis Overview
print(text.UNDERLINE + "Data Overview" + text.END)
# unusual characters [In-Text Citation: (Gosavi, 2024)]
english_regex = r"^[a-zA-Z0-9\s.,!?'\"\-/()&$#;:*+%]*$"
# finds cells that don't have english characters [In-Text Citation: (stites & Andy Hayden, 2013)]
print(f"Rows with non-english: {df[~df['Review'].str.contains(english_regex)].shape[0]}")

# tokenization
# retains only words and numbers, no symbols [In-Text Citation: (Bobbitt, 2022)]
sentences = df['Review'].str.replace(r'\W', ' ', regex=True)
sentences = [text.lower().split() for text in sentences]

# normalize sentences [In-Text Citation: (Jain, 2024)]
wl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
meaningful_sentences = []
meaningful_vocab = []
total_vocab = []
# this loop removes stopwords, lemmatizes the words, and makes a sequence of the words
for sentence in sentences:
   meaningful_words = []
   for word in sentence:
      if word not in total_vocab:
         total_vocab.append(word)
      # only retains non-stopwords
      if word not in stop_words:
         # lemmatizes the meaningful words
         lemmatized_word = wl.lemmatize(word)
         meaningful_words.append(lemmatized_word)
         # finds only unique words
         if lemmatized_word not in meaningful_vocab:
            meaningful_vocab.append(lemmatized_word)
   meaningful_sentences.append(meaningful_words)
print(f"Vocab of Words: {len(total_vocab)} words")
print(f"Vocab of Normalized Words: {len(meaningful_vocab)} words")

# find max sequence length
normalized_df = pd.DataFrame({
   'Review': map(" ".join, meaningful_sentences),
   'Length': map(len, meaningful_sentences),
   'Sentiment': df['Sentiment']
})
sequence_length = 25
print(f"Sequence Length: {sum(length <= sequence_length for length in normalized_df['Length'])} sentences have {sequence_length} words or less.")
plt.hist(normalized_df['Length'], bins=range(0,80,5), edgecolor='black')
plt.xlabel("Sentence Length")
plt.ylabel("Review Count")
plt.title("Length of Sentences")
plt.show()
# endregion

# region Data Preparation
print(text.UNDERLINE + "\nData Preparation" + text.END)
print("Proof of Tokenization:")
print(sentences[5])
print("Normalized Data:")
print(meaningful_sentences[5])
# split data
train_x, test_x, train_y, test_y = train_test_split(normalized_df['Review'], normalized_df['Sentiment'], test_size=0.2, random_state=101)
print(f"\nTraining Size: {train_x.shape[0]}, Testing Size: {test_x.shape[0]}")
# text vectorization and padding [In-Text Citation: (TensorFlow, 2024)]
vectorize_layer = TextVectorization(
   standardize='lower_and_strip_punctuation', 
   split='whitespace', 
   output_mode='int',
   output_sequence_length=sequence_length,
   )
# create a vocabulary based on training data
vectorize_layer.adapt(train_x)
vocab_size = vectorize_layer.vocabulary_size()
print(f"Vocabulary from Training Data: {vocab_size} words")
train_vector = vectorize_layer(train_x)
test_vector = vectorize_layer(test_x)
print("\nExample of Padded Sequence:")
tf.print(train_vector[0], summarize=-1)

training_x = pd.DataFrame(data=train_vector)
training_x.to_csv('train_x.csv', index=False)
testing_x = pd.DataFrame(data=test_vector)
testing_x.to_csv('test_x.csv', index=False)
training_y = pd.DataFrame(data=train_y)
training_y.to_csv('train_y.csv', index=False)
testing_y = pd.DataFrame(data=test_y)
testing_y.to_csv('test_y.csv', index=False)
# endregion

# region Build Model
print(text.UNDERLINE + "\nBuilding the Model" + text.END)
# early stopping [In-Text Citation: (Bhattbhatt, 2024)]
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
dimensions = 200
# switch to enable optimizing or to just run the best configured model,
# only enable if you have at least ~30 mins to spare
optimize = False
model_filename = 'Saved_Model.keras'
metrics_filename = 'Saved_Metrics.csv'
model = None
metrics_df = None
# checks if a generated model exists and loads it [In-Text Citation: (Anis, 2021)]
if os.path.isfile(model_filename) and os.path.isfile(metrics_filename):
   model = load_model(model_filename)
   # loads save history values for graphing
   metrics_df = pd.read_csv(metrics_filename)
else:
   if optimize:
      # 100 trials with 5 executions took ~30 mins to run
      tuner = RandomSearch(
         build_model,
         objective='val_accuracy',
         max_trials=100,
         executions_per_trial=5,
         directory='tuning_dir',
         project_name='my_model'
      )
      # Start the search for the best hyperparameters [In-Text Citation: (Rodrigo_V, 2021)]
      tuner.search(train_vector, train_y, epochs=20, batch_size=32, validation_data=(test_vector, test_y), callbacks=[early_stopping])
      # get best hyperparameters, not best model [In-Text Citation: (Kashyap, 2024)]
      best_hp = tuner.get_best_hyperparameters()[0]
      model = tuner.hypermodel.build(best_hp)
   else:
      # prebuilt model with best parameters from tuning [In-Text Citation: (Sharmasaravanan, 2024)]
      model = Sequential()
      model.add(Embedding(input_dim=vocab_size, output_dim=dimensions)) # embedding layer
      model.add(GlobalAveragePooling1D())
      model.add(Dropout(rate=0.2, seed=101))
      model.add(Dense(32, activation = "relu"))
      model.add(Dense(128, activation = "relu"))
      model.add(Dense(288, activation = "relu"))
      model.add(Dense(256, activation = "relu"))
      model.add(Dense(1, activation ='sigmoid'))  # output layer sigmoid b/c it is binary
      model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
   history = model.fit(train_vector, train_y, epochs=20, batch_size=32, validation_data=(test_vector, test_y), callbacks=[early_stopping])
   # saves history values so they can be loaded along with the saved model
   metrics_df = pd.DataFrame({
      'Accuracy': history.history['accuracy'],
      'Val_Accuracy': history.history['val_accuracy'],
      'Loss': history.history['loss'],
      'Val_Loss': history.history['val_loss'],
      'Epoch': range(1, len(history.epoch)+1),
      'Early_Stop': early_stopping.stopped_epoch - 1, # the -1 accounts for the patience of 3
   })
   metrics_df.to_csv(metrics_filename, index=False)
   # saves the model
   model.save(model_filename)
print()
print(model.summary())
# endregion

# region Evaluating Model 
print("\nModel Evaluation")
test_loss, test_acc = model.evaluate(test_vector, test_y)
print(f'Test Accuracy: {test_acc}')
print(f"Test Loss: {test_loss}")

plt.figure(figsize=(12, 5))
plt.xticks(range(1, len(metrics_df['Epoch'])+1))
if metrics_df['Early_Stop'][0] > 0: 
   plt.axvline(x=metrics_df['Early_Stop'][0], color="0.6", linestyle='dashed', label='Early Stop Triggered')
plt.plot(metrics_df['Epoch'], metrics_df['Accuracy'], label='Accuracy')
plt.plot(metrics_df['Epoch'], metrics_df['Val_Accuracy'], label='Validation Accuracy')
plt.plot(metrics_df['Epoch'], metrics_df['Loss'], label='Loss')
plt.plot(metrics_df['Epoch'], metrics_df['Val_Loss'], label='Validation Loss')
plt.title('Accuracy and Loss Scores Over Epochs')
plt.ylabel('Score')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# endregion
print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)