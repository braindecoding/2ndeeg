#!/usr/bin/env python3
# brain2image_cross_validation.py - Cross-subject validation using Brain2Image dataset

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from brain2image_dataset import Brain2ImageDataset

def load_trained_models():
    """Load our pre-trained ensemble models"""
    print("📂 Loading pre-trained ensemble models...")
    
    try:
        # Load traditional models
        traditional_models = joblib.load('traditional_models.pkl')
        
        # Load meta-model
        meta_model = joblib.load('meta_model.pkl')
        
        print("  ✅ Models loaded successfully")
        return traditional_models, meta_model
        
    except Exception as e:
        print(f"  ❌ Error loading models: {str(e)}")
        print("  ℹ️ Make sure you have run ensemble_model.py first to train the models")
        return None, None

def test_brain2image_subject(traditional_models, meta_model, features, labels):
    """Test a subject using our trained models on Brain2Image data"""
    print("  🧪 Testing Brain2Image classification...")
    
    try:
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get predictions from available traditional models
        predictions_list = []
        model_names = []
        
        if 'svm' in traditional_models and traditional_models['svm'] is not None:
            try:
                svm_proba = traditional_models['svm'].predict_proba(features_scaled)
                predictions_list.append(svm_proba)
                model_names.append('SVM')
                print("    ✅ SVM predictions obtained")
            except Exception as e:
                print(f"    ⚠️ SVM prediction failed: {str(e)}")
        
        if 'lr' in traditional_models and traditional_models['lr'] is not None:
            try:
                lr_proba = traditional_models['lr'].predict_proba(features_scaled)
                predictions_list.append(lr_proba)
                model_names.append('LR')
                print("    ✅ Logistic Regression predictions obtained")
            except Exception as e:
                print(f"    ⚠️ LR prediction failed: {str(e)}")
        
        if 'rf' in traditional_models and traditional_models['rf'] is not None:
            try:
                rf_proba = traditional_models['rf'].predict_proba(features_scaled)
                predictions_list.append(rf_proba)
                model_names.append('RF')
                print("    ✅ Random Forest predictions obtained")
            except Exception as e:
                print(f"    ⚠️ RF prediction failed: {str(e)}")
        
        if len(predictions_list) == 0:
            print("    ❌ No valid predictions from traditional models")
            return None
        
        # Individual model performances
        individual_results = {}
        for i, (proba, name) in enumerate(zip(predictions_list, model_names)):
            pred = np.argmax(proba, axis=1)
            acc = accuracy_score(labels, pred)
            individual_results[name] = {'accuracy': acc, 'predictions': pred}
            print(f"    📊 {name} individual accuracy: {acc:.4f}")
        
        # Combine predictions for meta-model
        if len(predictions_list) == 1:
            meta_features = predictions_list[0]
        else:
            meta_features = np.hstack(predictions_list)
        
        # Get final predictions from meta-model
        final_predictions = meta_model.predict(meta_features)
        final_proba = meta_model.predict_proba(meta_features)
        
        # Calculate ensemble accuracy
        ensemble_accuracy = accuracy_score(labels, final_predictions)
        
        print(f"    ✅ Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        # Detailed evaluation
        print("    📊 Ensemble Classification Report:")
        category_names = ['Faces', 'Objects']
        print(classification_report(labels, final_predictions, target_names=category_names))
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_predictions': final_predictions,
            'ensemble_probabilities': final_proba,
            'individual_results': individual_results
        }
        
    except Exception as e:
        print(f"    ❌ Error in Brain2Image testing: {str(e)}")
        return None

def cross_subject_brain2image_validation():
    """Perform cross-subject validation with Brain2Image dataset"""
    print("🚀 Cross-Subject Validation with Brain2Image Dataset")
    print("=" * 60)
    
    # Initialize Brain2Image dataset
    dataset = Brain2ImageDataset()
    
    # Create demo data if not exists
    if not os.path.exists(dataset.data_path):
        print("📁 Creating demo Brain2Image dataset...")
        dataset.create_demo_brain2image_data()
    
    # Load trained models
    traditional_models, meta_model = load_trained_models()
    
    if traditional_models is None or meta_model is None:
        print("❌ Cannot proceed without trained models")
        return
    
    # Test on each subject
    subjects_to_test = [1, 2, 3, 4]
    subject_results = {}
    
    for subject_id in subjects_to_test:
        print(f"\n🧪 Testing Subject {subject_id:02d} (Brain2Image)")
        print("-" * 50)
        
        # Load subject data
        eeg_data, labels, images = dataset.load_subject_data(subject_id)
        
        if eeg_data is None:
            print(f"  ⚠️ Skipping subject {subject_id} - data not available")
            continue
        
        # Extract Brain2Image specific features
        features = dataset.extract_brain2image_features(eeg_data)
        
        # Test with our models
        result = test_brain2image_subject(traditional_models, meta_model, features, labels)
        
        if result is not None:
            subject_results[subject_id] = {
                'ensemble_accuracy': result['ensemble_accuracy'],
                'ensemble_predictions': result['ensemble_predictions'],
                'ensemble_probabilities': result['ensemble_probabilities'],
                'individual_results': result['individual_results'],
                'true_labels': labels,
                'n_trials': len(labels),
                'images': images
            }
        else:
            print(f"    ❌ Testing failed for Subject {subject_id}")
    
    # Analysis and summary
    if len(subject_results) > 0:
        print(f"\n📊 Cross-Subject Brain2Image Validation Results")
        print("=" * 60)
        
        ensemble_accuracies = [result['ensemble_accuracy'] for result in subject_results.values()]
        mean_ensemble_accuracy = np.mean(ensemble_accuracies)
        std_ensemble_accuracy = np.std(ensemble_accuracies)
        
        print(f"Ensemble Mean Accuracy: {mean_ensemble_accuracy:.4f} ± {std_ensemble_accuracy:.4f}")
        print(f"Best Ensemble Accuracy: {np.max(ensemble_accuracies):.4f}")
        print(f"Worst Ensemble Accuracy: {np.min(ensemble_accuracies):.4f}")
        print(f"Number of subjects tested: {len(subject_results)}")
        
        # Individual model analysis
        all_individual_results = {}
        for subject_id, result in subject_results.items():
            for model_name, model_result in result['individual_results'].items():
                if model_name not in all_individual_results:
                    all_individual_results[model_name] = []
                all_individual_results[model_name].append(model_result['accuracy'])
        
        print(f"\nIndividual Model Performance:")
        print("-" * 40)
        for model_name, accuracies in all_individual_results.items():
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"{model_name}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Detailed results per subject
        print(f"\nDetailed Results per Subject:")
        print("-" * 40)
        for subject_id, result in subject_results.items():
            print(f"Subject {subject_id:02d}: Ensemble={result['ensemble_accuracy']:.4f} ({result['n_trials']} trials)")
            for model_name, model_result in result['individual_results'].items():
                print(f"  {model_name}: {model_result['accuracy']:.4f}")
        
        # Create comprehensive visualization
        create_brain2image_visualization(subject_results, mean_ensemble_accuracy)
        
        # Save results
        np.save('cross_subject_brain2image_results.npy', subject_results)
        print(f"\n📊 Detailed results saved as 'cross_subject_brain2image_results.npy'")
        
        return subject_results
    
    else:
        print("❌ No subjects were successfully tested")
        return None

def create_brain2image_visualization(subject_results, mean_ensemble_accuracy):
    """Create comprehensive visualization of Brain2Image validation results"""
    print("📊 Creating Brain2Image validation visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Extract data for plotting
    subjects = list(subject_results.keys())
    ensemble_accuracies = [subject_results[s]['ensemble_accuracy'] for s in subjects]
    
    # Subplot 1: Ensemble accuracy by subject
    plt.subplot(3, 3, 1)
    plt.bar([f"S{s:02d}" for s in subjects], ensemble_accuracies, color='steelblue')
    plt.axhline(y=mean_ensemble_accuracy, color='r', linestyle='--', 
                label=f'Mean: {mean_ensemble_accuracy:.3f}')
    plt.xlabel('Subject')
    plt.ylabel('Ensemble Accuracy')
    plt.title('Cross-Subject Ensemble Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Individual model comparison
    plt.subplot(3, 3, 2)
    all_individual_results = {}
    for subject_id, result in subject_results.items():
        for model_name, model_result in result['individual_results'].items():
            if model_name not in all_individual_results:
                all_individual_results[model_name] = []
            all_individual_results[model_name].append(model_result['accuracy'])
    
    model_names = list(all_individual_results.keys())
    model_means = [np.mean(all_individual_results[name]) for name in model_names]
    model_stds = [np.std(all_individual_results[name]) for name in model_names]
    
    plt.bar(model_names, model_means, yerr=model_stds, capsize=5, color='lightcoral')
    plt.axhline(y=mean_ensemble_accuracy, color='steelblue', linestyle='--', 
                label=f'Ensemble: {mean_ensemble_accuracy:.3f}')
    plt.xlabel('Model')
    plt.ylabel('Mean Accuracy')
    plt.title('Individual vs Ensemble Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Accuracy distribution
    plt.subplot(3, 3, 3)
    plt.hist(ensemble_accuracies, bins=5, alpha=0.7, edgecolor='black', color='steelblue')
    plt.axvline(x=mean_ensemble_accuracy, color='r', linestyle='--', 
                label=f'Mean: {mean_ensemble_accuracy:.3f}')
    plt.xlabel('Ensemble Accuracy')
    plt.ylabel('Frequency')
    plt.title('Accuracy Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Confusion matrix for best subject
    plt.subplot(3, 3, 4)
    best_subject_id = subjects[np.argmax(ensemble_accuracies)]
    best_predictions = subject_results[best_subject_id]['ensemble_predictions']
    best_labels = subject_results[best_subject_id]['true_labels']
    
    cm = confusion_matrix(best_labels, best_predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Best Subject S{best_subject_id:02d}')
    plt.colorbar()
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), 
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Faces', 'Objects'])
    plt.yticks([0, 1], ['Faces', 'Objects'])
    
    # Subplot 5: Performance vs original digit classification
    plt.subplot(3, 3, 5)
    original_accuracy = 0.83  # Our ensemble model accuracy on digit data
    comparison_data = ['Original\n(Digits)', 'Brain2Image\n(Faces/Objects)']
    comparison_acc = [original_accuracy, mean_ensemble_accuracy]
    colors = ['blue', 'orange']
    
    bars = plt.bar(comparison_data, comparison_acc, color=colors)
    plt.ylabel('Accuracy')
    plt.title('Domain Transfer Performance')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, comparison_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Subplot 6: Subject variability
    plt.subplot(3, 3, 6)
    subject_ids = list(subjects)
    plt.plot(subject_ids, ensemble_accuracies, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=mean_ensemble_accuracy, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Subject ID')
    plt.ylabel('Ensemble Accuracy')
    plt.title('Subject-to-Subject Variability')
    plt.grid(True, alpha=0.3)
    
    # Subplot 7-9: Sample images and predictions for best subject
    best_subject_data = subject_results[best_subject_id]
    images = best_subject_data['images']
    predictions = best_subject_data['ensemble_predictions']
    true_labels = best_subject_data['true_labels']
    
    # Show 3 examples
    for i in range(3):
        plt.subplot(3, 3, 7 + i)
        
        # Find a correctly classified example
        correct_indices = np.where(predictions == true_labels)[0]
        if len(correct_indices) > i:
            idx = correct_indices[i]
            plt.imshow(images[idx], cmap='gray')
            category_name = 'Face' if true_labels[idx] == 0 else 'Object'
            plt.title(f'Correct: {category_name}')
        else:
            plt.text(0.5, 0.5, 'No example', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Example N/A')
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cross_subject_brain2image_validation.png', dpi=300, bbox_inches='tight')
    print("  ✅ Visualization saved as 'cross_subject_brain2image_validation.png'")
    
    plt.close()

def main():
    """Main function"""
    print("🧪 Cross-Subject Validation with Brain2Image Dataset")
    print("=" * 60)
    
    # Run cross-subject validation
    results = cross_subject_brain2image_validation()
    
    if results:
        ensemble_accuracies = [result['ensemble_accuracy'] for result in results.values()]
        mean_accuracy = np.mean(ensemble_accuracies)
        
        print("\n✅ Cross-subject validation completed successfully!")
        print(f"📊 Mean ensemble accuracy: {mean_accuracy:.4f}")
        
        # Interpretation for Brain2Image
        print(f"\n🔍 Brain2Image Classification Analysis:")
        if mean_accuracy > 0.7:
            print("  ✅ Excellent generalization for visual perception tasks")
            print("  💡 Model successfully transfers from digit imagery to face/object perception")
        elif mean_accuracy > 0.6:
            print("  ⚠️ Good generalization with some domain adaptation needed")
            print("  💡 Consider fine-tuning for visual perception tasks")
        else:
            print("  ❌ Limited generalization - significant domain gap")
            print("  💡 Recommend task-specific training or transfer learning")
        
        # Domain transfer analysis
        original_accuracy = 0.83
        transfer_ratio = mean_accuracy / original_accuracy
        
        print(f"\n📈 Domain Transfer Analysis:")
        print(f"  Original task accuracy: {original_accuracy:.3f}")
        print(f"  Brain2Image task accuracy: {mean_accuracy:.3f}")
        print(f"  Transfer ratio: {transfer_ratio:.3f}")
        
        if transfer_ratio > 0.8:
            print("  ✅ Excellent domain transfer capability")
        elif transfer_ratio > 0.6:
            print("  ⚠️ Good domain transfer with room for improvement")
        else:
            print("  ❌ Poor domain transfer - consider domain adaptation")
        
        print(f"\n📝 Recommendations for Brain2Image Research:")
        print("  1. Implement domain adaptation techniques")
        print("  2. Use visual perception specific feature extraction")
        print("  3. Consider multi-task learning approaches")
        print("  4. Explore image reconstruction from EEG features")
        print("  5. Validate on real Brain2Image dataset")
        
    else:
        print("\n❌ Cross-subject validation failed!")

if __name__ == "__main__":
    main()
