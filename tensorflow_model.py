import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

class TensorFlowModel(tf.keras.Model):
    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(4, activation='softmax')
        
    def call(self, inputs):
        x = self.pool(self.conv1(inputs))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

def get_tensorflow_datagen(data_dir, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, val_generator

def train_tensorflow(model, data_dir, batch_size, epochs, learning_rate):
    train_generator, val_generator = get_tensorflow_datagen(data_dir, batch_size)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )
    
    print('Finished Training')
    

    # Store the trained model
    os.makedirs('models', exist_ok=True)
    model.save('models/Fabrice_model.tensorflow')
    print("✅ Modèle TensorFlow sauvegardé dans : models/Fabrice_model.tensorflow")
    
    # === Display and store the plots ===
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

   # === Training (Accuracy + Loss) ===
    #plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-o', label='Accuracy')
    plt.plot(epochs_range, loss, 'g-o', label='Loss')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.savefig('tensorflow_training_graph.png')  # Sauvegarde avant affichage
    plt.show()
