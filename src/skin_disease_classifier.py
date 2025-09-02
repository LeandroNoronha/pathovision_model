# ----------------------------------------------------------------------------
# Created By  : Leandro Noronha da Silva
# Created Date: 26/08/2025
# version ='1.0'
# ---------------------------------------------------------------------------
""" PATHOVISION """
# ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class SkinDiseaseClassifier:
    ### ***************************
    ### TODO Image size for dataset
    ### ***************************
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        self.num_classes = num_classes # Number of disease classes
        self.input_shape = input_shape # Input image format
        self.model = None
        self.history = None
        
    def build_model(self):
        # Loads pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Frezze initial layers for transfer learning
        for layer in base_model.layers[:-10]:
            layer.trainable = False
            
        # Adds custom layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile final model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        ### *************************************
        ### TODO F1 score needs to be implemented
        ### *************************************
        
        return self.model
    
    def prepare_data_generators(self, train_dir, val_dir, batch_size=32):
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Data generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50):
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                'best_skin_disease_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Model training
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        return self.history
    
    def plot_training_history(self):
        if self.history is None:
            print("Model not trained yet!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_image(self, image_path, class_names):
        # Loads and process image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return class_names[predicted_class], confidence
    
    def evaluate_model(self, test_generator, class_names):

        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Classification report
        print("Classification report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion matrix')
        plt.ylabel('Real Class')
        plt.xlabel('Predicted Class')
        plt.show()

def main():
    # Defines the 6 skin diseases classes
    class_names = [
        # Fungal Infections
        'Tinea',
        'Ringworm',
        'Candidiasis',
        # Inflammatory Conditions
        'Psoriasis',
        'Eczema',
        'Atopic Dermatitis'
    ]
    
    num_classes = len(class_names)
    
    # Initialize the classifier
    classifier = SkinDiseaseClassifier(num_classes=num_classes)
    
    # Build the model
    model = classifier.build_model()
    print(f"Model created with {model.count_params()} parameters")

if __name__ == "__main__":
    main()