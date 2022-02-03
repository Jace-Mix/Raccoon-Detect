from python import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os

# Argument Construction
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--plot', type=str, default='plot.png', help='path to output loss/accuracy plot')
args = vars(ap.parse_args())

# Initialization
INIT_LR = 1e-4
EPOCHS = 5
BS = 32

# Load images
print('Retrieving images...')
imagePaths = list(paths.list_images(config.BASE_PATH))
data = []
labels = []

# Loop over image paths
for imagePath in imagePaths:
    
    # Get labels from filenames
    label = imagePath.split(os.path.sep)[-2]
    
    # Load image and prepare
    image = load_img(imagePath, target_size=config.INPUT_DIMS)
    image = img_to_array(image)
    image = preprocess_input(image)
    
    # update data and labels
    data.append(image)
    labels.append(label)

# Convert to numpy
data = np.array(data, dtype='float32')
labels = np.array(labels)

# Change labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Obtain test and train data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Create image generator
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Retrieve MobileNetV2 without top head
orig_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Replace top head
headModel = orig_model.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

# Place top head onto original model
model = Model(inputs=orig_model.input, outputs=headModel)

# Print summary of network model
model.summary()

# Disable training on layers
for layer in orig_model.layers:
    layer.trainable = False
    
# Compile Model
print('Compiling model...')
opt = Adam(lr=INIT_LR)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train head of network
print('Training Head...')
history = model.fit(aug.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS, validation_data=(testX, testY), validation_steps=len(testX) // BS, epochs=EPOCHS)

# Predict using head
print('Evaluating network...')
predIdxs = model.predict(testX, batch_size=BS)

# Find largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Print classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Save model
print('Saving detector model...')
model.save(config.MODEL_PATH, save_format='h5')

# Save label encoder
print('Saving label encoder...')
f = open(config.ENCODER_PATH, 'wb')
f.write(pickle.dumps(lb))
f.close()

# Plot train loss and accuracy
plt.plot(np.arange(0, EPOCHS), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, EPOCHS), history.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, EPOCHS), history.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, EPOCHS), history.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper right')
plt.savefig(args['plot'])