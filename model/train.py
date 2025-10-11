import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = 'dataset'  # Relative to model folder

def create_model(num_classes=4):
    """
    Create a CNN model for banana classification
    Using transfer learning with MobileNetV2
    """
    # Load pre-trained MobileNetV2 (trained on ImageNet)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers on top
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_data():
    """
    Load and prepare the dataset
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'valid'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_generator, val_generator

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("📊 Training history saved as 'training_history.png'")

def main():
    print("🍌 Banana Death Predictor - Model Training")
    print("=" * 50)
    
    # Load data
    print("\n📁 Loading dataset...")
    train_generator, val_generator = load_data()
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Classes: {train_generator.class_indices}")
    
    # Create model
    print("\n🏗️ Building model...")
    model = create_model(num_classes=len(train_generator.class_indices))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # Train model
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("This will take some time. Go grab a coffee! ☕")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    train_loss, train_acc = model.evaluate(train_generator)
    val_loss, val_acc = model.evaluate(val_generator)
    
    print(f"\n✅ Training Accuracy: {train_acc*100:.2f}%")
    print(f"✅ Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save model
    model_path = 'saved_models/banana_model.keras'
    os.makedirs('saved_models', exist_ok=True)
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n🎉 Training completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()