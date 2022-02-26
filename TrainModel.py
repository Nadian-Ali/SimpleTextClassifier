# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:37:48 2021

@author: Nadian
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

batch_size = 4
seed = 42
folder_name =  'TextClassification'

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    folder_name, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    # class_names = ['Yes','No','Maybe'],
    seed=seed)




raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    folder_name, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    # class_names = ['Yes','No','Maybe'],
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    folder_name, 
    batch_size=batch_size)


max_features = 200
sequence_length = 5

vectorize_layer = tf.keras.layers.TextVectorization(
    # standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)   

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  # layers.Dense(8,activation='sigmoid'),
  layers.Dense(1)])

model.summary()



model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=.1),
              # metrics = ['accuracy'])
               metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


epochs = 60
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=.1),
              # metrics = ['accuracy'])
               metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)




examples = [
  "yes",
  
]

A = export_model.predict(examples)

# export_model.save('D:/TextClassification/clsModel')

tf.keras.models.save_model(
    export_model,
    'D:/TextClassification/mymodel',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)