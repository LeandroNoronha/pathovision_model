# ----------------------------------------------------------------------------
# Created By  : Leandro Noronha da Silva
# Created Date: 26/08/2025
# version ='2.0.0'
# ---------------------------------------------------------------------------
""" PATHOVISION """
# ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import os

class SkinDiseaseClassifier:
    def __init__(self, num_classes, input_shape=(500, 500, 3)):
        self.num_classes = num_classes # Number of disease classes
        self.input_shape = input_shape # Input image format
        self.model = None
        self.history = None
        
    def build_model(self):
        # Load pre-trained EfficientNetB2
        base_model = EfficientNetB2(
            weights='imagenet',
            include_top=False,
            input_shape=(500, 500, 3)
        )
        
        # Freeze initial layers for transfer learning
        for layer in base_model.layers[:-10]:
            layer.trainable = False
            
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Custom F1-Score metric for multi-class
        def f1_metric(y_true, y_pred):
            def recall_m(y_true, y_pred):
                TP = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
                Positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
                recall = TP / (Positives + tf.keras.backend.epsilon())
                return recall
            
            def precision_m(y_true, y_pred):
                TP = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
                Pred_Positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
                precision = TP / (Pred_Positives + tf.keras.backend.epsilon())
                return precision
            
            precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))
        
        # Compile model with accuracy and F1-score metrics
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', f1_metric]
        )
        
        return self.model
    
    def prepare_data_generators(self, train_dir, val_dir, batch_size=32):
        # Data augmentation for training with EfficientNet preprocessing
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only EfficientNet preprocessing for validation
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        # Data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb'
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50):
        # Training callbacks
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
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))        

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return class_names[predicted_class], confidence
    
    def calculate_metrics(self, y_true, y_pred, class_names):
        """
        Compute essential evaluation metrics with explicit formulas:
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - Sensitivity (Recall) = TP / (TP + FN)
        - Specificity = TN / (TN + FP)
        - Precision = TP / (TP + FP)
        - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        """
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Metrics per class
        metrics = {}
        
        # For each class, compute TP, TN, FP, FN
        for i, class_name in enumerate(class_names):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - TP - FN - FP
            
            # Metrics using formulas
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            metrics[class_name] = {
                'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Precision': precision,
                'F1-Score': f1
            }
        
        # Global metrics (weighted by support where applicable)
        global_accuracy = np.sum([metrics[cls]['TP'] for cls in class_names]) / len(y_true)
        global_precision = precision_score(y_true, y_pred, average='weighted')
        global_recall = recall_score(y_true, y_pred, average='weighted')
        global_f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics['Global'] = {
            'Accuracy': global_accuracy,
            'Precision': global_precision,
            'Sensitivity': global_recall,
            'F1-Score': global_f1
        }
        
        return metrics, cm
    
    def plot_metrics_comparison(self, metrics, class_names):
        """Plot comparative bar charts of metrics per class."""
        
        # Extract per-class metrics
        classes = [cls for cls in class_names]
        accuracies = [metrics[cls]['Accuracy'] for cls in classes]
        sensitivities = [metrics[cls]['Sensitivity'] for cls in classes]
        specificities = [metrics[cls]['Specificity'] for cls in classes]
        precisions = [metrics[cls]['Precision'] for cls in classes]
        f1_scores = [metrics[cls]['F1-Score'] for cls in classes]
        
        # Subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Metrics per Class', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0,0].bar(classes, accuracies, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Accuracy per Class')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Sensitivity (Recall)
        axes[0,1].bar(classes, sensitivities, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Sensitivity (Recall) per Class')
        axes[0,1].set_ylabel('Sensitivity')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Specificity
        axes[0,2].bar(classes, specificities, color='lightcoral', alpha=0.7)
        axes[0,2].set_title('Specificity per Class')
        axes[0,2].set_ylabel('Specificity')
        axes[0,2].set_ylim(0, 1)
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Precision
        axes[1,0].bar(classes, precisions, color='gold', alpha=0.7)
        axes[1,0].set_title('Precision per Class')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1,1].bar(classes, f1_scores, color='mediumpurple', alpha=0.7)
        axes[1,1].set_title('F1-Score per Class')
        axes[1,1].set_ylabel('F1-Score')
        axes[1,1].set_ylim(0, 1)
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Combined comparison
        x = np.arange(len(classes))
        width = 0.15
        
        axes[1,2].bar(x - 2*width, accuracies, width, label='Accuracy', alpha=0.8)
        axes[1,2].bar(x - width, sensitivities, width, label='Sensitivity', alpha=0.8)
        axes[1,2].bar(x, specificities, width, label='Specificity', alpha=0.8)
        axes[1,2].bar(x + width, precisions, width, label='Precision', alpha=0.8)
        axes[1,2].bar(x + 2*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        axes[1,2].set_title('Comparison of All Metrics')
        axes[1,2].set_ylabel('Score')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(classes, rotation=45)
        axes[1,2].legend()
        axes[1,2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, test_generator, class_names):
        """Plot ROC curves and AUC for each class."""
        
        # Predictions and true labels (one-hot)
        predictions = self.model.predict(test_generator)
        y_true = tf.keras.utils.to_categorical(test_generator.classes, num_classes=self.num_classes)
        
        plt.figure(figsize=(12, 8))
        
        # Class-wise ROC
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Random classifier diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curves per Class')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate_model(self, test_generator, class_names):
        """Comprehensive model evaluation with metrics, confusion matrix, and ROC curves."""
        
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Compute metrics using implemented formulas
        metrics, cm = self.calculate_metrics(y_true, y_pred, class_names)
        
        # Detailed metrics report
        print("="*80)
        print("DETAILED METRICS REPORT")
        print("="*80)
        
        # Per-class metrics
        for class_name in class_names:
            print(f"\n{class_name.upper()}:")
            print(f"   TP: {metrics[class_name]['TP']}, TN: {metrics[class_name]['TN']}")
            print(f"   FP: {metrics[class_name]['FP']}, FN: {metrics[class_name]['FN']}")
            print(f"   Accuracy:      {metrics[class_name]['Accuracy']:.4f}")
            print(f"   Sensitivity:   {metrics[class_name]['Sensitivity']:.4f}")
            print(f"   Specificity:   {metrics[class_name]['Specificity']:.4f}")
            print(f"   Precision:     {metrics[class_name]['Precision']:.4f}")
            print(f"   F1-Score:      {metrics[class_name]['F1-Score']:.4f}")
        
        # Global metrics
        print(f"\nGLOBAL METRICS:")
        print(f"   Global Accuracy:   {metrics['Global']['Accuracy']:.4f}")
        print(f"   Global Precision:  {metrics['Global']['Precision']:.4f}")
        print(f"   Global Sensitivity:{metrics['Global']['Sensitivity']:.4f}")
        print(f"   Global F1-Score:   {metrics['Global']['F1-Score']:.4f}")
        
        # Classification report (sklearn)
        print("\nCLASSIFICATION REPORT (sklearn):")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Plots
        self.plot_metrics_comparison(metrics, class_names)
        self.plot_confusion_matrix(cm, class_names)
        self.plot_roc_curves(test_generator, class_names)
        
        return metrics
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot an enhanced confusion matrix with counts and row-wise percentages."""
        plt.figure(figsize=(12, 10))
        
        # Row-wise percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Combined annotations (count + percent)
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
        
        sns.heatmap(
            cm, annot=annotations, fmt='', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Number of Samples'}
        )
        
        plt.title('Confusion Matrix\n(Count and Row Percentage)', fontsize=14, fontweight='bold')
        plt.ylabel('True Class', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

def main():
    base_dir = "/home/lns/repos/research_project-unisinos/dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "valid")
    test_dir = os.path.join(base_dir, "test")
    
    class_names = sorted(os.listdir(train_dir))
    num_classes = len(class_names)
    
    print(f"Detected classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # Start the classifier
    classifier = SkinDiseaseClassifier(num_classes=num_classes, input_shape=(500, 500, 3))
    
    # Build model
    model = classifier.build_model()
    print(f"Model created with {model.count_params()} parameters")
    
    # Generate train and validation
    train_gen, val_gen = classifier.prepare_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=32
    )
    
    # Train
    history = classifier.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=50
    )
    
    # Plot train x validation
    classifier.plot_training_history()
    
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
if __name__ == "__main__":
    main()