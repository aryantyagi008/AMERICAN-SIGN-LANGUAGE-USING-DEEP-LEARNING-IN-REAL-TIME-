# asl_model_training.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 15

def load_data():
    train_dir = 'asl_dataset/asl_alphabet_train'
    test_dir = 'asl_dataset/asl_alphabet_test'
    
    # First, get complete list of classes from training data
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    NUM_CLASSES = len(train_classes)
    print(f"Training classes detected: {NUM_CLASSES} - {train_classes}")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    
    # Validation data should not be augmented
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow training images - using all classes
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        classes=train_classes)
    
    # Flow test images - using same class list as training
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        classes=train_classes)  # Force same classes as training
    
    # Verify we have same number of classes in both generators
    assert len(train_generator.class_indices) == len(test_generator.class_indices), \
        f"Mismatched classes: train has {len(train_generator.class_indices)}, test has {len(test_generator.class_indices)}"
    
    class_names = list(train_generator.class_indices.keys())
    return train_generator, test_generator, class_names, NUM_CLASSES

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def train_model():
    # Load data
    train_generator, test_generator, class_names, NUM_CLASSES = load_data()
    
    print(f"\nClass names: {class_names}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {test_generator.samples}")
    
    # Model
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    model = create_model(input_shape, NUM_CLASSES)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'asl_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max')
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True)
    
    # Calculate steps
    train_steps = train_generator.samples // BATCH_SIZE
    test_steps = test_generator.samples // BATCH_SIZE
    
    # Training
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=test_generator,
        validation_steps=test_steps,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop])
    
    # Save class names
    np.save('class_names.npy', class_names)
    
    # Evaluation
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'\nTest accuracy: {test_acc:.4f}')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()

if __name__ == '__main__':
    # Enable GPU memory growth if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    train_model()