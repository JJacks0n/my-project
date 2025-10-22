"""
Predictive Provenance: A Large Action Model for Anticipating Sensemaking Trajectories

This module implements a comprehensive system for predicting user behavior patterns
in sensemaking activities using provenance data from interaction sequences.

Based on:
- Provectories: Embedding-based Analysis of Interaction Provenance Data
- Projection Space Explorer data
- Sensemaking trajectory analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
import umap
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available. Some clustering features will be disabled.")
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

class ProvenanceDataProcessor:
    """Process and prepare provenance data for model training"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def load_synthetic_data(self, n_samples=1000, n_features=50):
        """Generate synthetic provenance data for demonstration"""
        np.random.seed(42)
        
        # Generate interaction sequences
        sequences = []
        labels = []
        
        for i in range(n_samples):
            # Generate a sequence of interactions
            seq_length = np.random.randint(10, 50)
            sequence = []
            
            # Simulate different types of user behavior patterns
            behavior_type = np.random.choice(['explorer', 'analyst', 'focused', 'random'])
            
            if behavior_type == 'explorer':
                # Explorer: broad, diverse interactions
                for j in range(seq_length):
                    interaction = {
                        'timestamp': j,
                        'action_type': np.random.choice(['zoom', 'pan', 'filter', 'select', 'brush']),
                        'data_dimension': np.random.randint(0, 10),
                        'interaction_intensity': np.random.uniform(0.3, 1.0),
                        'spatial_position': np.random.uniform(0, 1, 2),
                        'data_density': np.random.uniform(0.1, 0.9)
                    }
                    sequence.append(interaction)
                    
            elif behavior_type == 'analyst':
                # Analyst: systematic, methodical interactions
                for j in range(seq_length):
                    interaction = {
                        'timestamp': j,
                        'action_type': np.random.choice(['filter', 'select', 'compare']),
                        'data_dimension': j % 5,  # More systematic
                        'interaction_intensity': np.random.uniform(0.6, 1.0),
                        'spatial_position': np.array([0.5, 0.5]) + np.random.normal(0, 0.1, 2),
                        'data_density': np.random.uniform(0.5, 0.9)
                    }
                    sequence.append(interaction)
                    
            elif behavior_type == 'focused':
                # Focused: concentrated on specific areas
                focus_area = np.random.uniform(0, 1, 2)
                for j in range(seq_length):
                    interaction = {
                        'timestamp': j,
                        'action_type': np.random.choice(['zoom', 'select', 'brush']),
                        'data_dimension': np.random.randint(0, 3),  # Limited dimensions
                        'interaction_intensity': np.random.uniform(0.7, 1.0),
                        'spatial_position': focus_area + np.random.normal(0, 0.05, 2),
                        'data_density': np.random.uniform(0.6, 1.0)
                    }
                    sequence.append(interaction)
                    
            else:  # random
                for j in range(seq_length):
                    interaction = {
                        'timestamp': j,
                        'action_type': np.random.choice(['zoom', 'pan', 'filter', 'select', 'brush', 'reset']),
                        'data_dimension': np.random.randint(0, 10),
                        'interaction_intensity': np.random.uniform(0.1, 0.8),
                        'spatial_position': np.random.uniform(0, 1, 2),
                        'data_density': np.random.uniform(0.1, 0.7)
                    }
                    sequence.append(interaction)
            
            sequences.append(sequence)
            labels.append(behavior_type)
        
        return sequences, labels
    
    def extract_features(self, sequences):
        """Extract meaningful features from interaction sequences"""
        features = []
        
        for seq in sequences:
            if len(seq) == 0:
                continue
                
            # Temporal features
            timestamps = [interaction['timestamp'] for interaction in seq]
            duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
            interaction_rate = len(seq) / max(duration, 1)
            
            # Action type features
            action_types = [interaction['action_type'] for interaction in seq]
            unique_actions = len(set(action_types))
            action_diversity = unique_actions / len(action_types) if len(action_types) > 0 else 0
            
            # Spatial features
            positions = np.array([interaction['spatial_position'] for interaction in seq])
            spatial_variance = np.var(positions, axis=0).mean() if len(positions) > 1 else 0
            spatial_range = np.max(positions, axis=0) - np.min(positions, axis=0)
            spatial_coverage = np.prod(spatial_range)
            
            # Intensity features
            intensities = [interaction['interaction_intensity'] for interaction in seq]
            avg_intensity = np.mean(intensities)
            intensity_variance = np.var(intensities)
            
            # Data dimension features
            dimensions = [interaction['data_dimension'] for interaction in seq]
            dimension_diversity = len(set(dimensions)) / len(dimensions) if len(dimensions) > 0 else 0
            
            # Data density features
            densities = [interaction['data_density'] for interaction in seq]
            avg_density = np.mean(densities)
            density_variance = np.var(densities)
            
            # Sequence pattern features
            action_transitions = []
            for i in range(len(action_types) - 1):
                action_transitions.append(f"{action_types[i]}_to_{action_types[i+1]}")
            transition_diversity = len(set(action_transitions)) / len(action_transitions) if len(action_transitions) > 0 else 0
            
            # Create feature vector
            feature_vector = [
                len(seq),  # sequence length
                duration,  # total duration
                interaction_rate,  # interactions per time unit
                unique_actions,  # number of unique actions
                action_diversity,  # action diversity ratio
                spatial_variance,  # spatial variance
                spatial_coverage,  # spatial coverage
                avg_intensity,  # average interaction intensity
                intensity_variance,  # intensity variance
                dimension_diversity,  # dimension diversity
                avg_density,  # average data density
                density_variance,  # density variance
                transition_diversity,  # transition diversity
            ]
            
            # Add action type frequencies
            action_freq = {}
            for action in ['zoom', 'pan', 'filter', 'select', 'brush', 'compare', 'reset']:
                action_freq[action] = action_types.count(action) / len(action_types)
                feature_vector.append(action_freq[action])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_feature_names(self):
        """Create descriptive names for features"""
        base_features = [
            'sequence_length', 'duration', 'interaction_rate', 'unique_actions',
            'action_diversity', 'spatial_variance', 'spatial_coverage',
            'avg_intensity', 'intensity_variance', 'dimension_diversity',
            'avg_density', 'density_variance', 'transition_diversity'
        ]
        
        action_features = [
            'zoom_freq', 'pan_freq', 'filter_freq', 'select_freq',
            'brush_freq', 'compare_freq', 'reset_freq'
        ]
        
        return base_features + action_features

class ProvenancePredictor:
    """Main class for predicting sensemaking trajectories"""
    
    def __init__(self):
        self.processor = ProvenanceDataProcessor()
        self.models = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def prepare_data(self, sequences, labels):
        """Prepare data for training"""
        # Extract features
        features = self.processor.extract_features(sequences)
        self.feature_names = self.processor.create_feature_names()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        return features_scaled, labels_encoded, label_encoder
    
    def train_models(self, X, y):
        """Train multiple models for comparison"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Compute class weights to mitigate imbalance
        try:
            from sklearn.utils.class_weight import compute_class_weight
            import numpy as np
            classes = np.unique(y_train)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
        except Exception:
            class_weight_dict = None
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight_dict)
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_score = gb_model.score(X_test, y_test)
        
        self.models = {
            'Random Forest': {'model': rf_model, 'score': rf_score},
            'Gradient Boosting': {'model': gb_model, 'score': gb_score}
        }
        
        return X_test, y_test
    
    def predict_behavior(self, sequences):
        """Predict behavior type for new sequences"""
        features = self.processor.extract_features(sequences)
        features_scaled = self.scaler.transform(features)
        
        predictions = {}
        for name, model_info in self.models.items():
            pred = model_info['model'].predict(features_scaled)
            pred_proba = model_info['model'].predict_proba(features_scaled)
            predictions[name] = {
                'predictions': pred,
                'probabilities': pred_proba
            }
        
        return predictions
    
    def analyze_feature_importance(self):
        """Analyze feature importance across models"""
        importance_data = []
        
        for model_name, model_info in self.models.items():
            if hasattr(model_info['model'], 'feature_importances_'):
                importances = model_info['model'].feature_importances_
                for i, importance in enumerate(importances):
                    importance_data.append({
                        'feature': self.feature_names[i],
                        'importance': importance,
                        'model': model_name
                    })
        
        return pd.DataFrame(importance_data)

class SensemakingTrajectoryAnalyzer:
    """Analyze and visualize sensemaking trajectories"""
    
    def __init__(self):
        self.trajectory_graph = nx.DiGraph()
        
    def build_trajectory_graph(self, sequences, labels):
        """Build a graph representation of sensemaking trajectories"""
        for i, (seq, label) in enumerate(zip(sequences, labels)):
            # Add nodes for each interaction
            for j, interaction in enumerate(seq):
                node_id = f"user_{i}_step_{j}"
                # Create node attributes without conflicting keys
                node_attrs = {k: v for k, v in interaction.items() if k not in ['user_id', 'step', 'behavior_type']}
                node_attrs.update({
                    'user_id': i,
                    'step': j, 
                    'behavior_type': label
                })
                self.trajectory_graph.add_node(node_id, **node_attrs)
                
                # Add edges between consecutive interactions
                if j > 0:
                    prev_node_id = f"user_{i}_step_{j-1}"
                    self.trajectory_graph.add_edge(prev_node_id, node_id)
    
    def find_common_patterns(self):
        """Find common interaction patterns in the trajectory graph"""
        # Find frequent subgraphs
        patterns = {}
        
        # Analyze action sequences
        action_sequences = []
        for node in self.trajectory_graph.nodes():
            if 'action_type' in self.trajectory_graph.nodes[node]:
                action_sequences.append(self.trajectory_graph.nodes[node]['action_type'])
        
        # Find common 3-grams
        from collections import Counter
        ngrams = []
        for i in range(len(action_sequences) - 2):
            ngram = tuple(action_sequences[i:i+3])
            ngrams.append(ngram)
        
        pattern_counts = Counter(ngrams)
        common_patterns = pattern_counts.most_common(10)
        
        return common_patterns
    
    def visualize_trajectories(self, sequences, labels, max_sequences=50):
        """Visualize sensemaking trajectories"""
        fig = go.Figure()
        
        colors = {'explorer': 'red', 'analyst': 'blue', 'focused': 'green', 'random': 'orange'}
        
        for i, (seq, label) in enumerate(zip(sequences[:max_sequences], labels[:max_sequences])):
            if len(seq) == 0:
                continue
                
            # Extract positions
            positions = np.array([interaction['spatial_position'] for interaction in seq])
            timestamps = [interaction['timestamp'] for interaction in seq]
            
            # Create trajectory line
            fig.add_trace(go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='lines+markers',
                line=dict(color=colors.get(label, 'gray'), width=2),
                marker=dict(size=6),
                name=f'User {i} ({label})',
                showlegend=False,
                hovertemplate=f'User {i} ({label})<br>Step: %{{text}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                text=timestamps
            ))
        
        fig.update_layout(
            title='Sensemaking Trajectories Visualization',
            xaxis_title='Spatial Position X',
            yaxis_title='Spatial Position Y',
            width=800,
            height=600
        )
        
        return fig
    
    def create_behavior_heatmap(self, sequences, labels):
        """Create a heatmap showing behavior patterns"""
        # Extract features for heatmap
        processor = ProvenanceDataProcessor()
        features = processor.extract_features(sequences)
        feature_names = processor.create_feature_names()
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        df['behavior_type'] = labels
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix

def main():
    """Main function to demonstrate the predictive provenance system"""
    print("Predictive Provenance: A Large Action Model for Anticipating Sensemaking Trajectories")
    print("=" * 80)
    
    # Initialize components
    predictor = ProvenancePredictor()
    analyzer = SensemakingTrajectoryAnalyzer()
    
    # Generate synthetic data
    print("Generating synthetic provenance data...")
    sequences, labels = predictor.processor.load_synthetic_data(n_samples=500)
    print(f"Generated {len(sequences)} interaction sequences")
    
    # Prepare data
    print("Preparing data for training...")
    X, y, label_encoder = predictor.prepare_data(sequences, labels)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Train models
    print("Training predictive models...")
    X_test, y_test = predictor.train_models(X, y)
    
    # Display model performance
    print("\nModel Performance:")
    for name, model_info in predictor.models.items():
        print(f"{name}: {model_info['score']:.3f}")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_df = predictor.analyze_feature_importance()
    top_features = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
    print("Top 10 most important features:")
    for feature, importance in top_features.items():
        print(f"  {feature}: {importance:.3f}")
    
    # Build trajectory graph
    print("\nBuilding trajectory graph...")
    analyzer.build_trajectory_graph(sequences, labels)
    print(f"Graph has {analyzer.trajectory_graph.number_of_nodes()} nodes and {analyzer.trajectory_graph.number_of_edges()} edges")
    
    # Find common patterns
    print("\nFinding common interaction patterns...")
    patterns = analyzer.find_common_patterns()
    print("Top 5 common interaction patterns:")
    for pattern, count in patterns[:5]:
        print(f"  {pattern}: {count} occurrences")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Trajectory visualization
    trajectory_fig = analyzer.visualize_trajectories(sequences, labels, max_sequences=30)
    trajectory_fig.show()
    
    # Behavior heatmap
    analyzer.create_behavior_heatmap(sequences, labels)
    
    # Test predictions on new data
    print("\nTesting predictions on new sequences...")
    test_sequences, test_labels = predictor.processor.load_synthetic_data(n_samples=50)
    predictions = predictor.predict_behavior(test_sequences)
    
    # Display prediction results
    for model_name, pred_info in predictions.items():
        pred_labels = label_encoder.inverse_transform(pred_info['predictions'])
        true_labels = label_encoder.inverse_transform(predictor.processor.label_encoder.transform(test_labels))
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"{model_name} prediction accuracy: {accuracy:.3f}")
    
    print("\nPredictive Provenance system demonstration completed!")
    print("The system can now predict user behavior patterns in sensemaking activities.")

if __name__ == "__main__":
    main()
