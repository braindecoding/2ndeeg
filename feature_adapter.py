#!/usr/bin/env python3
# feature_adapter.py - Adapter to match feature dimensions between datasets

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureAdapter:
    """Adapter to match feature dimensions between different datasets"""
    
    def __init__(self, target_features=2560):
        self.target_features = target_features
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_selector = None
        self.is_fitted = False
        
    def fit_transform(self, features, labels=None):
        """Fit adapter and transform features to target dimension"""
        print(f"🔧 Adapting features from {features.shape[1]} to {self.target_features}...")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        current_features = features_scaled.shape[1]
        
        if current_features == self.target_features:
            print("  ✅ Features already match target dimension")
            self.is_fitted = True
            return features_scaled
        
        elif current_features < self.target_features:
            # Need to expand features
            print(f"  🔧 Expanding features from {current_features} to {self.target_features}")
            expanded_features = self._expand_features(features_scaled)
            self.is_fitted = True
            return expanded_features
            
        else:
            # Need to reduce features
            print(f"  🔧 Reducing features from {current_features} to {self.target_features}")
            
            if labels is not None:
                # Use feature selection if labels available
                self.feature_selector = SelectKBest(f_classif, k=min(self.target_features, current_features))
                reduced_features = self.feature_selector.fit_transform(features_scaled, labels)
                
                # If still need more reduction, use PCA
                if reduced_features.shape[1] > self.target_features:
                    self.pca = PCA(n_components=self.target_features)
                    final_features = self.pca.fit_transform(reduced_features)
                else:
                    final_features = reduced_features
            else:
                # Use PCA only
                self.pca = PCA(n_components=self.target_features)
                final_features = self.pca.fit_transform(features_scaled)
            
            self.is_fitted = True
            return final_features
    
    def transform(self, features):
        """Transform new features using fitted adapter"""
        if not self.is_fitted:
            raise ValueError("Adapter must be fitted first")
        
        # Standardize
        features_scaled = self.scaler.transform(features)
        
        current_features = features_scaled.shape[1]
        
        if current_features == self.target_features:
            return features_scaled
        
        elif current_features < self.target_features:
            return self._expand_features(features_scaled)
        
        else:
            # Reduce features
            if self.feature_selector is not None:
                reduced_features = self.feature_selector.transform(features_scaled)
                if self.pca is not None:
                    return self.pca.transform(reduced_features)
                else:
                    return reduced_features
            elif self.pca is not None:
                return self.pca.transform(features_scaled)
            else:
                return features_scaled[:, :self.target_features]
    
    def _expand_features(self, features):
        """Expand features to target dimension"""
        current_features = features.shape[1]
        n_samples = features.shape[0]
        
        # Calculate how many times to repeat and how many extra features needed
        repeat_times = self.target_features // current_features
        extra_features = self.target_features % current_features
        
        expanded_parts = []
        
        # Repeat original features
        for i in range(repeat_times):
            # Add small noise to make repeated features slightly different
            noise_scale = 0.01 * (i + 1)
            noise = np.random.normal(0, noise_scale, features.shape)
            expanded_parts.append(features + noise)
        
        # Add extra features if needed
        if extra_features > 0:
            # Use polynomial features or interactions
            extra_part = self._create_polynomial_features(features, extra_features)
            expanded_parts.append(extra_part)
        
        # Combine all parts
        expanded_features = np.hstack(expanded_parts)
        
        print(f"    ✅ Expanded to {expanded_features.shape[1]} features")
        return expanded_features
    
    def _create_polynomial_features(self, features, n_extra):
        """Create polynomial/interaction features"""
        n_samples, n_features = features.shape
        
        # Create polynomial features (squares, products, etc.)
        poly_features = []
        
        # Add squared features
        for i in range(min(n_extra, n_features)):
            poly_features.append(features[:, i] ** 2)
        
        # Add interaction features if still need more
        remaining = n_extra - len(poly_features)
        if remaining > 0:
            for i in range(n_features):
                for j in range(i+1, n_features):
                    if len(poly_features) >= n_extra:
                        break
                    poly_features.append(features[:, i] * features[:, j])
                if len(poly_features) >= n_extra:
                    break
        
        # Add random combinations if still need more
        while len(poly_features) < n_extra:
            i, j = np.random.choice(n_features, 2, replace=False)
            poly_features.append(features[:, i] * features[:, j] + np.random.normal(0, 0.01, n_samples))
        
        # Stack and return only the needed number
        poly_array = np.column_stack(poly_features[:n_extra])
        return poly_array

def test_feature_adapter():
    """Test the feature adapter"""
    print("🧪 Testing Feature Adapter")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    
    # Small feature set (like visual EEG)
    small_features = np.random.randn(100, 284)
    labels = np.random.randint(0, 2, 100)
    
    # Large feature set (like original dataset)
    large_features = np.random.randn(100, 3000)
    
    # Test expanding small features
    print("\n1. Testing feature expansion:")
    adapter1 = FeatureAdapter(target_features=2560)
    expanded = adapter1.fit_transform(small_features, labels)
    print(f"   Input: {small_features.shape} -> Output: {expanded.shape}")
    
    # Test reducing large features
    print("\n2. Testing feature reduction:")
    adapter2 = FeatureAdapter(target_features=2560)
    reduced = adapter2.fit_transform(large_features, labels)
    print(f"   Input: {large_features.shape} -> Output: {reduced.shape}")
    
    # Test transform on new data
    print("\n3. Testing transform on new data:")
    new_small = np.random.randn(50, 284)
    new_expanded = adapter1.transform(new_small)
    print(f"   New input: {new_small.shape} -> Output: {new_expanded.shape}")
    
    print("\n✅ Feature adapter test completed!")

if __name__ == "__main__":
    test_feature_adapter()
