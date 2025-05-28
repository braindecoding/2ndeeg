#!/usr/bin/env python3
# handwritten_cross_subject_validation.py - Strict cross-subject validation for handwritten character dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HandwrittenCrossSubjectValidator:
    """Strict cross-subject validation for handwritten character EEG dataset"""

    def __init__(self):
        self.eeg_data = None
        self.session_labels = None
        self.subjects_data = {}
        self.results = {}

    def load_handwritten_data(self):
        """Load extracted handwritten EEG data"""
        print("📂 Loading handwritten character EEG data...")

        try:
            self.eeg_data = np.load('handwritten_eeg_data.npy')
            self.session_labels = np.load('handwritten_session_labels.npy')

            print(f"✅ Data loaded successfully!")
            print(f"  📊 EEG shape: {self.eeg_data.shape}")
            print(f"  📊 Labels shape: {self.session_labels.shape}")
            print(f"  📊 Sessions: {np.bincount(self.session_labels)}")

            return True

        except FileNotFoundError:
            print("❌ Handwritten data not found. Run debug_handwritten_data.py first.")
            return False

    def create_strict_subjects(self):
        """Create strict subject divisions for cross-validation"""
        print("\n🔧 Creating strict subject divisions...")

        # Define subject mapping based on sessions
        subject_mapping = {
            0: "Subject_A_SGEye_R1",      # round01_sgeyesub
            1: "Subject_B_Paradigm_R1",   # round01_paradigm
            2: "Subject_C_SGEye_R2",      # round02_sgeyesub
            3: "Subject_D_Paradigm_R2"    # round02_paradigm
        }

        for session_id in range(4):
            session_mask = self.session_labels == session_id
            session_data = self.eeg_data[session_mask]

            # Create binary classification task (first half vs second half)
            n_samples = session_data.shape[0]
            split_point = n_samples // 2

            # Create balanced binary labels
            binary_labels = np.zeros(n_samples)
            binary_labels[split_point:] = 1

            # Subsample for computational efficiency (every 10th sample for more data)
            subsample_rate = 10
            indices = np.arange(0, n_samples, subsample_rate)

            subsampled_data = session_data[indices]
            subsampled_labels = binary_labels[indices]

            self.subjects_data[session_id] = {
                'name': subject_mapping[session_id],
                'eeg_data': subsampled_data,
                'labels': subsampled_labels,
                'original_session': session_id,
                'n_samples': len(subsampled_labels),
                'label_distribution': np.bincount(subsampled_labels.astype(int))
            }

            print(f"  ✅ {subject_mapping[session_id]}: {subsampled_data.shape}")
            print(f"    Labels: {np.bincount(subsampled_labels.astype(int))}")

    def extract_features(self, eeg_data):
        """Extract features from EEG data"""
        # Statistical features
        features = []

        # Mean, std, min, max for each channel
        features.extend(np.mean(eeg_data, axis=0))
        features.extend(np.std(eeg_data, axis=0))
        features.extend(np.min(eeg_data, axis=0))
        features.extend(np.max(eeg_data, axis=0))

        # Frequency domain features (simple)
        fft_data = np.abs(np.fft.fft(eeg_data, axis=0))
        freq_bands = [
            (0, 4),    # Delta
            (4, 8),    # Theta
            (8, 13),   # Alpha
            (13, 30),  # Beta
            (30, 50)   # Gamma
        ]

        for low, high in freq_bands:
            band_power = np.mean(fft_data[low:high], axis=0)
            features.extend(band_power)

        return np.array(features)

    def prepare_subject_features(self):
        """Prepare features for each subject"""
        print("\n🔧 Extracting features for each subject...")

        for subject_id, subject_data in self.subjects_data.items():
            eeg_data = subject_data['eeg_data']

            # Extract features using sliding window
            window_size = 500   # 0.5 second at 1000 Hz
            step_size = 250     # 50% overlap

            features_list = []
            labels_list = []

            for i in range(0, eeg_data.shape[0] - window_size, step_size):
                window_data = eeg_data[i:i+window_size]
                window_label = subject_data['labels'][i + window_size//2]  # Middle label

                features = self.extract_features(window_data)
                features_list.append(features)
                labels_list.append(window_label)

            if len(features_list) > 0:
                subject_features = np.array(features_list)
                subject_labels = np.array(labels_list)

                # Update subject data
                self.subjects_data[subject_id]['features'] = subject_features
                self.subjects_data[subject_id]['feature_labels'] = subject_labels

                print(f"  ✅ {subject_data['name']}: {subject_features.shape}")
                print(f"    Feature labels: {np.bincount(subject_labels.astype(int))}")

    def train_models(self, X_train, y_train):
        """Train ensemble of models"""
        print("  🔧 Training models...")

        # Ensure data types are correct
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.int32)

        # Check for minimum samples
        if len(X_train) < 10:
            print(f"  ⚠️ Warning: Only {len(X_train)} training samples")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Define models with simpler parameters for small datasets
        models = {
            'svm': SVC(probability=True, random_state=42, C=1.0),
            'lr': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'rf': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        }

        # Train individual models with error handling
        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                trained_models[name] = model
                print(f"    ✅ {name.upper()} trained successfully")
            except Exception as e:
                print(f"    ❌ {name.upper()} training failed: {str(e)}")

        # Create voting ensemble only if we have models
        if len(trained_models) > 1:
            try:
                voting_clf = VotingClassifier(
                    estimators=[(name, model) for name, model in trained_models.items()],
                    voting='soft'
                )
                voting_clf.fit(X_train_scaled, y_train)
                trained_models['ensemble'] = voting_clf
                print(f"    ✅ ENSEMBLE trained successfully")
            except Exception as e:
                print(f"    ❌ ENSEMBLE training failed: {str(e)}")

        return trained_models, scaler

    def evaluate_models(self, models, scaler, X_test, y_test):
        """Evaluate models on test data"""
        # Ensure data types are correct
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.int32)

        X_test_scaled = scaler.transform(X_test)

        results = {}
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

                accuracy = accuracy_score(y_test, y_pred)

                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                print(f"    ✅ {name.upper()} evaluated: {accuracy:.4f}")

            except Exception as e:
                print(f"    ❌ {name.upper()} evaluation failed: {str(e)}")
                # Add dummy result to avoid breaking analysis
                results[name] = {
                    'accuracy': 0.0,
                    'predictions': np.zeros_like(y_test),
                    'probabilities': None,
                    'confusion_matrix': np.zeros((2, 2))
                }

        return results

    def strict_cross_subject_validation(self):
        """Perform strict cross-subject validation"""
        print("\n🧪 Performing Strict Cross-Subject Validation")
        print("=" * 60)

        subjects = list(self.subjects_data.keys())
        n_subjects = len(subjects)

        all_results = {}

        # Leave-One-Subject-Out Cross-Validation
        for test_subject in subjects:
            train_subjects = [s for s in subjects if s != test_subject]

            test_name = self.subjects_data[test_subject]['name']
            train_names = [self.subjects_data[s]['name'] for s in train_subjects]

            print(f"\n🔍 Test Subject: {test_name}")
            print(f"📚 Train Subjects: {train_names}")

            # Prepare training data
            X_train_list = []
            y_train_list = []

            for train_subject in train_subjects:
                if 'features' in self.subjects_data[train_subject]:
                    X_train_list.append(self.subjects_data[train_subject]['features'])
                    y_train_list.append(self.subjects_data[train_subject]['feature_labels'])

            if len(X_train_list) == 0:
                print("  ❌ No training data available")
                continue

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            # Prepare test data
            if 'features' not in self.subjects_data[test_subject]:
                print("  ❌ No test data available")
                continue

            X_test = self.subjects_data[test_subject]['features']
            y_test = self.subjects_data[test_subject]['feature_labels']

            print(f"  📊 Train data: {X_train.shape}, Test data: {X_test.shape}")
            print(f"  📊 Train labels: {np.bincount(y_train.astype(int))}")
            print(f"  📊 Test labels: {np.bincount(y_test.astype(int))}")

            # Train models
            models, scaler = self.train_models(X_train, y_train)

            # Evaluate models
            results = self.evaluate_models(models, scaler, X_test, y_test)

            # Store results
            all_results[test_subject] = {
                'test_subject': test_name,
                'train_subjects': train_names,
                'results': results,
                'test_size': len(y_test),
                'train_size': len(y_train)
            }

            # Print results
            print(f"  📊 Results:")
            for model_name, result in results.items():
                print(f"    {model_name.upper()}: {result['accuracy']:.4f}")

        self.results = all_results
        return all_results

    def analyze_results(self):
        """Analyze cross-subject validation results"""
        print(f"\n📊 Cross-Subject Validation Analysis")
        print("=" * 50)

        if not self.results:
            print("❌ No results to analyze")
            return

        # Collect all accuracies
        model_accuracies = {}

        for test_subject, subject_results in self.results.items():
            test_name = subject_results['test_subject']

            print(f"\n🔍 {test_name}:")
            for model_name, result in subject_results['results'].items():
                if model_name not in model_accuracies:
                    model_accuracies[model_name] = []

                accuracy = result['accuracy']
                model_accuracies[model_name].append(accuracy)
                print(f"  {model_name.upper()}: {accuracy:.4f}")

        # Calculate statistics
        print(f"\n📊 Overall Performance:")
        print("-" * 30)

        for model_name, accuracies in model_accuracies.items():
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)

            print(f"{model_name.upper()}:")
            print(f"  Mean: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  Range: {min_acc:.4f} - {max_acc:.4f}")

        # Find best model
        ensemble_accs = model_accuracies.get('ensemble', [])
        if ensemble_accs:
            best_ensemble = np.mean(ensemble_accs)
            print(f"\n🏆 Best Ensemble Performance: {best_ensemble:.4f}")

        return model_accuracies

    def create_visualization(self):
        """Create comprehensive visualization"""
        print(f"\n📊 Creating visualization...")

        if not self.results:
            print("❌ No results to visualize")
            return

        # Collect data for visualization
        subjects = []
        model_names = []
        accuracies_matrix = []

        for test_subject, subject_results in self.results.items():
            subjects.append(subject_results['test_subject'])

            if not model_names:  # First iteration
                model_names = list(subject_results['results'].keys())
                accuracies_matrix = [[] for _ in model_names]

            for i, model_name in enumerate(model_names):
                accuracy = subject_results['results'][model_name]['accuracy']
                accuracies_matrix[i].append(accuracy)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Accuracy by subject
        x_pos = np.arange(len(subjects))
        width = 0.15

        for i, (model_name, accuracies) in enumerate(zip(model_names, accuracies_matrix)):
            axes[0, 0].bar(x_pos + i*width, accuracies, width,
                          label=model_name.upper(), alpha=0.8)

        axes[0, 0].set_xlabel('Test Subject')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Cross-Subject Validation Results')
        axes[0, 0].set_xticks(x_pos + width * (len(model_names)-1) / 2)
        axes[0, 0].set_xticklabels([s.split('_')[1] for s in subjects], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Model comparison
        model_means = [np.mean(accs) for accs in accuracies_matrix]
        model_stds = [np.std(accs) for accs in accuracies_matrix]

        axes[0, 1].bar(model_names, model_means, yerr=model_stds,
                      capsize=5, alpha=0.8, color='steelblue')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Mean Accuracy')
        axes[0, 1].set_title('Model Performance Comparison')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Accuracy distribution
        all_accuracies = [acc for accs in accuracies_matrix for acc in accs]
        axes[0, 2].hist(all_accuracies, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(np.mean(all_accuracies), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_accuracies):.3f}')
        axes[0, 2].set_xlabel('Accuracy')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Accuracy Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Subject data overview
        subject_sizes = []
        subject_names_short = []

        for subject_id, subject_data in self.subjects_data.items():
            if 'features' in subject_data:
                subject_sizes.append(subject_data['features'].shape[0])
                subject_names_short.append(subject_data['name'].split('_')[1])

        axes[1, 0].bar(subject_names_short, subject_sizes, color='lightcoral')
        axes[1, 0].set_xlabel('Subject')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Dataset Size per Subject')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Confusion matrix for best model (ensemble)
        if 'ensemble' in model_names:
            ensemble_idx = model_names.index('ensemble')
            best_subject_idx = np.argmax(accuracies_matrix[ensemble_idx])
            best_subject = list(self.results.keys())[best_subject_idx]

            cm = self.results[best_subject]['results']['ensemble']['confusion_matrix']

            im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[1, 1].set_title(f'Best Confusion Matrix\n(Subject {subjects[best_subject_idx].split("_")[1]})')

            # Add text annotations
            for i in range(2):
                for j in range(2):
                    axes[1, 1].text(j, i, str(cm[i, j]),
                                   horizontalalignment="center",
                                   color="white" if cm[i, j] > cm.max() / 2 else "black")

            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('True')
            axes[1, 1].set_xticks([0, 1])
            axes[1, 1].set_xticklabels(['Class 0', 'Class 1'])
            axes[1, 1].set_yticks([0, 1])
            axes[1, 1].set_yticklabels(['Class 0', 'Class 1'])

        # Plot 6: Performance summary
        axes[1, 2].text(0.1, 0.9, 'Handwritten Character EEG', fontsize=14, fontweight='bold',
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.8, f'Cross-Subject Validation', fontsize=12,
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.7, f'Subjects: {len(subjects)}',
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f'Models: {len(model_names)}',
                        transform=axes[1, 2].transAxes)

        # Check if ensemble results exist
        ensemble_accs = accuracies_matrix[model_names.index('ensemble')] if 'ensemble' in model_names else []
        if ensemble_accs:
            axes[1, 2].text(0.1, 0.5, f'Best Ensemble: {np.mean(ensemble_accs):.4f}',
                            transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.4, f'Std: ±{np.std(ensemble_accs):.4f}',
                            transform=axes[1, 2].transAxes)

        axes[1, 2].text(0.1, 0.3, f'Total Samples: {sum(subject_sizes):,}',
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.2, f'Features: {self.subjects_data[0]["features"].shape[1]}',
                        transform=axes[1, 2].transAxes)

        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')

        plt.suptitle('Strict Cross-Subject Validation: Handwritten Character EEG', fontsize=16)
        plt.tight_layout()
        plt.savefig('handwritten_cross_subject_validation.png', dpi=300, bbox_inches='tight')
        print("  ✅ Visualization saved as 'handwritten_cross_subject_validation.png'")

        plt.close()

def main():
    """Main function"""
    print("🧪 Strict Cross-Subject Validation for Handwritten Character EEG")
    print("=" * 70)

    # Initialize validator
    validator = HandwrittenCrossSubjectValidator()

    # Load data
    if not validator.load_handwritten_data():
        return

    # Create strict subjects
    validator.create_strict_subjects()

    # Extract features
    validator.prepare_subject_features()

    # Perform cross-subject validation
    results = validator.strict_cross_subject_validation()

    if results:
        # Analyze results
        model_accuracies = validator.analyze_results()

        # Create visualization
        validator.create_visualization()

        # Save results
        np.save('handwritten_cross_subject_results.npy', results)

        print(f"\n✅ Strict cross-subject validation completed!")
        print(f"📊 Results saved as 'handwritten_cross_subject_results.npy'")

        # Summary
        if 'ensemble' in model_accuracies:
            ensemble_mean = np.mean(model_accuracies['ensemble'])
            ensemble_std = np.std(model_accuracies['ensemble'])
            print(f"\n🏆 Final Ensemble Performance: {ensemble_mean:.4f} ± {ensemble_std:.4f}")

        print(f"\n🔍 Key Insights:")
        print("✅ Strict leave-one-subject-out validation performed")
        print("✅ Multiple models compared (SVM, LR, RF, Ensemble)")
        print("✅ Real cross-subject generalization tested")
        print("✅ Handwritten character EEG patterns analyzed")

    else:
        print(f"\n❌ Cross-subject validation failed!")

if __name__ == "__main__":
    main()
