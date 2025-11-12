"""
PATHOVISION Model for UNISINOS Research Project

Created By  : Leandro Noronha da Silva
Created Date: 26/08/2025
version = '4.0.0'
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import os

class PathovisionClassifier:
    """
    This class implements the transfer learning with EfficientNetB2 
    for classify dermatologic images from dataset
    """
    
    def __init__(self, num_classes, input_shape=(500, 500, 3)):
        """
        This definition initializes the skin disease classifier
        """
        self.num_classes = num_classes # number of disease classes
        self.input_shape = input_shape # input image format
        self.model = None
        self.history = None
   
    def model_pathovision_construct(self):
        """
        This definition construct the PATHOVISION model
        """
        # load pre-trained EfficientNetB2
        base_model = EfficientNetB2(
            include_top=False, 
            weights='imagenet', # pre-training weights for fast conversion
            input_tensor=None,
            input_shape=self.input_shape, # (500, 500, 3) shape
            pooling=None,
            classifier_activation=None, # unnecessary because include_top = false
            name="efficientnetb2",
        )
        
        # freeze initial layers for transfer learning
        for layer in base_model.layers[:-10]:
            layer.trainable = False # only the last 10 layers trainable
            
        # add custom classification head
        x = base_model.output # gets the output from base_model (EfficientNetB2)
        x = GlobalAveragePooling2D()(x) # converts from 4D to 2D
        x = Dense(512, activation='relu')(x) # 512 neurons with rectified linear unit
        x = Dropout(0.5)(x) # turn off 50% from neuros randomly
        x = Dense(256, activation='relu')(x) # 256 neurons with rectified linear unit
        x = Dropout(0.3)(x) # turn off 30% from neuros randomly
        predictions = Dense(self.num_classes, activation='softmax')(x) # final classifier layer
        
        # create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        def f1_metric(y_true, y_pred):
            """
            This definition calculates the F1-Score metric for 
            multi-class
            """

            def recall_metric(y_true, y_pred):
                """
                This definition calculates the recall metric
                """
                # calculates the True Positives (TP)
                TP = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
                # calculates the real positives (True Positives (TP) + False Negatives (FN))
                Positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
                # calculates the recall metric (TP / (TP + FN))
                recall = TP / (Positives + tf.keras.backend.epsilon())
                return recall
            
            def precision_metric(y_true, y_pred):
                """
                This definition calculates the precision metric
                """
                # calculates the True Positives (TP)
                TP = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
                # calculates the positive predictions (True Positives (TP) + False Positives (FP))
                Pred_Positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
                # calculates the precision metric (TP / (TP + FP))
                precision = TP / (Pred_Positives + tf.keras.backend.epsilon())
                return precision
            
            # calculate precision and recall
            precision = precision_metric(y_true, y_pred)
            recall = recall_metric(y_true, y_pred)

            # return F1-score
            return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        
        # compile model with accuracy and F1-score metrics
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001), # Adaptive Moment Estimation at 1e-4
            loss='categorical_crossentropy',
            metrics=[   # performance monitoring
                'accuracy', 
                f1_metric
            ]
        )
        
        return self.model
    
    def prepare_data_generators(self, train_dir, val_dir, batch_size=32):
        """
        This definition prepares data generators for:
            -> training (applies data augmentation) 
            -> validation (applies only normalization)
        """

        # data augmentation for training with EfficientNet preprocessing
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input, # normalizes pixels to the range expected by the network
            rotation_range=20, # rotates randomly the image to +/- 20 degrees
            width_shift_range=0.2, # shift images horizontally up to 20% of their size
            height_shift_range=0.2, # shift images vertically up to 20% of their size
            horizontal_flip=True, # 50% chance of mirroring the image
            zoom_range=0.2, # simulates different camera distances.
            shear_range=0.2, # simulates different photo capture angles
            fill_mode='nearest' # copies the nearest pixel (avoids artificial black borders)
        )
        
        # only EfficientNet preprocessing for validation
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        # train data generator with augmentation
        train_generator = train_datagen.flow_from_directory(
            train_dir, 
            target_size=self.input_shape[:2], # guarantees resizing of all images to (500, 500) pixels
            batch_size=batch_size, # loads 32 images at a time (batch_size).
            class_mode='categorical', # labels in one-hot encoding format
            color_mode='rgb' # loads images in RGB (Red, Green, Blue)
        )
        # validation data generator without augmentation
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb'
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50):
        """
        This definition is responsible for train the PATHOVISION model
        """

        # Training callbacks
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss', # monitors the validation loss
                factor=0.2, # reduces the Learning Rate (LR) in 20%
                patience=5, # wait 5 epochs without improvement before reduce LR
                min_lr=1e-7 # minimum learning rate
            ),
            ModelCheckpoint(
                'best_skin_disease_model.h5', # output model archive
                monitor='val_accuracy', # monitors the validation accuracy
                save_best_only=True, # save only when val_accuracy improves
                mode='max' # better when higher val_accuracy
            )
        ]
        
        # Model training from Keras
        self.history = self.model.fit(
            train_generator, # Training data generator (with augmentation)
            epochs=epochs, # number of ephocs (50)
            validation_data=val_generator, # Evaluate the model after each season
            callbacks=callbacks # apply the callbacks defined previously
        )
        
        return self.history

    def show_top_confusions(self, test_generator, class_names, k=5):
        """
        This definition is responsible for display the 5 (five) more 
        confident model errors
        """

        filepaths = np.array(test_generator.filepaths) # get the full paths of all images in the test set
        # get the true labels (real classes) for each image (e.g. 0=tinea, 1=candidiasis, ...)
        y_true = test_generator.classes 
        y_score = self.model.predict(test_generator) # makes predictions on all test images
        y_pred = np.argmax(y_score, axis=1) # get the predicted class for each image
        confidences = y_score.max(axis=1) # get the confidence of each prediction

        # find the indices of the images where the model made a mistake
        errors_idx = np.where(y_pred != y_true)[0]
        if len(errors_idx) == 0:
            print("!!! No errors in the test (PATHOVISION model perfect 100%) !!!")
            return

        # sort errors by confidence (highest to lowest)
        sorted_err = errors_idx[np.argsort(-confidences[errors_idx])]

        print(f"Top {min(k, len(sorted_err))} more confident errors:")       
        for i in sorted_err[:k]: # loop through the first k most confident errors
            true_cls = class_names[y_true[i]]
            pred_cls = class_names[y_pred[i]]
            conf = confidences[i]
            print(f"[{conf:.3f}] True: {true_cls} | Pred: {pred_cls} | Path: {filepaths[i]}")
    
    def predict_image(self, image_path, class_names):
        """
        This definition is responsible for make a prediction on a 
        single image
        """
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.input_shape[:2]  # guarantees resizing of all images to (500, 500) pixels
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) # converts the PIL image (500, 500) into a 3D NumPy array (RGB)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0)) # applies specific preprocessing from EfficientNet        

        predictions = self.model.predict(img_array) # performs the forward pass through the neural network
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] # accesses the probability value in the predicted class index
        
        return class_names[predicted_class], confidence
    
    def calculate_metrics(self, y_true, y_pred, class_names):
        """
        This definition is responsible for compute essential 
        evaluation metrics
        """

        cm = confusion_matrix(y_true, y_pred) # creates the confusion matrix
        metrics = {} # initializes a dictionary to store the metrics calculated for each class
        
        # For each class, compute TP, TN, FP, FN
        for i, class_name in enumerate(class_names):
            TP = cm[i, i] # TP -> true positives
            FN = np.sum(cm[i, :]) - TP # FN -> false negatives
            FP = np.sum(cm[:, i]) - TP # FP -> false positives
            TN = np.sum(cm) - TP - FN - FP # TN -> true negatives
            
            # calculates all metrics based on formulas
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # adds the calculated metrics per class to the dictionary
            metrics[class_name] = {
                'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
                'Accuracy': accuracy, # accuracy = (TP + TN) / (TP + TN + FP + FN)
                'Sensitivity': sensitivity, # sensitivity = TP / (TP + FN)
                'Specificity': specificity, # specificity = TN / (TN + FP)
                'Precision': precision, # precision = TP / (TP + FP)
                'F1-Score': f1 # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
            }
        
        # Global metrics (weighted by support where applicable)
        global_accuracy = np.sum([metrics[cls]['TP'] for cls in class_names]) / len(y_true)
        global_precision = precision_score(y_true, y_pred, average='weighted')
        global_recall = recall_score(y_true, y_pred, average='weighted')
        global_f1 = f1_score(y_true, y_pred, average='weighted')
        
        # adds global metrics to the dictionary
        metrics['Global'] = {
            'Accuracy': global_accuracy,
            'Precision': global_precision,
            'Sensitivity': global_recall,
            'F1-Score': global_f1
        }
        
        return metrics, cm
    
    def evaluate_model(self, test_generator, class_names):
        """
        This definition is responsible to evaluates the model
        performance on the test set
        """
        
        predictions = self.model.predict(test_generator) # processes all images in the test set through the model
        y_pred = np.argmax(predictions, axis=1) # selects the class with the highest probability
        y_true = test_generator.classes # access the actual labels of the images
        
        # compute metrics using implemented formulas
        metrics, cm = self.calculate_metrics(y_true, y_pred, class_names)

        # detailed metrics report visualization
        print(" >>> DETAILED VIEW OF GLOBAL AND PER CLASS METRICS <<<")
        
        # metrics per class
        for class_name in class_names:
            print(f"\n{class_name.upper()}:")
            print(f"   TP: {metrics[class_name]['TP']}, TN: {metrics[class_name]['TN']}")
            print(f"   FP: {metrics[class_name]['FP']}, FN: {metrics[class_name]['FN']}")
            print(f"   Accuracy:      {metrics[class_name]['Accuracy']:.4f}")
            print(f"   Sensitivity:   {metrics[class_name]['Sensitivity']:.4f}")
            print(f"   Specificity:   {metrics[class_name]['Specificity']:.4f}")
            print(f"   Precision:     {metrics[class_name]['Precision']:.4f}")
            print(f"   F1-Score:      {metrics[class_name]['F1-Score']:.4f}")
        
        # global metrics
        print(f"\nGLOBAL METRICS:")
        print(f"   Global Accuracy:   {metrics['Global']['Accuracy']:.4f}")
        print(f"   Global Precision:  {metrics['Global']['Precision']:.4f}")
        print(f"   Global Sensitivity:{metrics['Global']['Sensitivity']:.4f}")
        print(f"   Global F1-Score:   {metrics['Global']['F1-Score']:.4f}")
        
        # classification report (sklearn)
        print("\nCLASSIFICATION REPORT (sklearn):")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        return metrics

def main():
    # HardCoded dataset directory
    base_dir = "/home/lns/repos/research_project-unisinos/dataset"
    train_dir = os.path.join(base_dir, "train") # defines the base path for the train dataset directory
    val_dir = os.path.join(base_dir, "valid") # defines the base path for the validation dataset directory
    test_dir = os.path.join(base_dir, "test") # defines the base path for the test dataset directory
    
    # automatic class detection
    class_names = sorted(os.listdir(train_dir))
    num_classes = len(class_names)
    
    # display the class detection
    print(f"Detected classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # start the classifier
    classifier = PathovisionClassifier(num_classes=num_classes, input_shape=(500, 500, 3))
    
    # build model architecture
    model = classifier.model_pathovision_construct()
    print(f"Model created with {model.count_params()} parameters")
    
    # generate train and validation
    train_gen, val_gen = classifier.prepare_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=32
    )
    
    # model train
    history = classifier.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=50
    )
    
    # Test generator
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=classifier.input_shape[:2],
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    # Evaluation on the Test Set
    metrics = classifier.evaluate_model(test_gen, class_names)

    # Show top confusions
    classifier.show_top_confusions(test_gen, class_names, k=10)

if __name__ == "__main__":
    main()