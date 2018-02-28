# Convolutional Neural Network

# Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_yaml

# Initialising the CNN
classifier = Sequential()

# Step 1 Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# 32 feature detectors with a 3x3 feature detector -> 32 feature maps
# input shape of the pictures has to be uniform -> size drastically influences algorithm completion time

# Step 2 Pooling (reducing size of feature map) and divide them by two
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding second convolutional layer, input_shape is only needed for first convolutional layer
#classifier.add(Convolution2D(32, (3, 3), activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 Flattening the layer after pooling step
classifier.add(Flatten())

# Step 4 Full connection of the network (start of classic ANN)
classifier.add(Dense(output_dim=128, activation='relu'))
# output_dim should be between input and output layers - if not known a power of 2 is good
# output layer
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# compiling the model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image preprocessing (augmentation)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), # equal dimensions expected by CNN
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250, # numbers of images in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=63) # numbers of test set

# look at some cnn parameters

# serialize model to YAML
classifier_yaml = classifier.to_yaml()
with open("cats_dogs_classifier.yaml", "w") as yaml_file:
    yaml_file.write(classifier_yaml)
# serialize weights to HDF5
classifier.save_weights("cats_dogs_weights.h5")
print("Saved model to disk")

# save weights in variable explorer
for layer in classifier.layers:
      weights = layer.get_weights()

# load YAML and create model
yaml_file = open('cats_dogs_classifier.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# get weights from loaded model
for layer in loaded_model.layers:
      weights_loaded = layer.get_weights()

























