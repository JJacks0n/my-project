"""
Real Data Loader for Projection Space Explorer CSV

This module loads and processes the actual CSV data from the projection-space-explorer
repository to create real interaction sequences for our predictive model.
"""

import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ProjectionSpaceDataLoader:
    """Load and process real data from projection-space-explorer CSV files"""
    
    def __init__(self):
        self.csv_urls = {
            'track_stories': 'https://raw.githubusercontent.com/jku-vds-lab/projection-space-explorer/master/notebooks/conny_all_track_stories/all_trrack_stories_20210609.csv',
            'lc0_all': 'https://raw.githubusercontent.com/jku-vds-lab/projection-space-explorer/master/notebooks/lc0%20misc/alphazero_vs_stockfish_all.csv',
            'lc0_merged': 'https://raw.githubusercontent.com/jku-vds-lab/projection-space-explorer/master/notebooks/lc0%20misc/alphazero_vs_stockfish_merged.csv',
            'test_cluster': 'https://raw.githubusercontent.com/jku-vds-lab/projection-space-explorer/master/notebooks/testcluster.csv'
        }
        
    def download_csv_data(self, dataset_name='track_stories'):
        """Download CSV data from GitHub"""
        print(f"Downloading {dataset_name} dataset...")
        
        if dataset_name not in self.csv_urls:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        url = self.csv_urls[dataset_name]
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to local file
            filename = f"real_data_{dataset_name}.csv"
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded {filename} ({len(response.content)} bytes)")
            return filename
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return None
    
    def load_track_stories_data(self, filename):
        """Load and process the track stories CSV data"""
        print(f"Loading track stories data from {filename}...")
        
        df = pd.read_csv(filename)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Basic statistics
        print(f"Users: {df['user'].nunique()}")
        print(f"Tasks: {df['task'].nunique()}")
        print(f"Lines: {df['line'].nunique()}")
        
        return df

    def load_testcluster_data(self, filename):
        """Load and adapt testcluster CSV (no user/task columns)."""
        print(f"Loading testcluster data from {filename}...")
        df = pd.read_csv(filename)
        # create synthetic identifiers to group into sequences
        if 'line' not in df.columns:
            df['line'] = 0
        df['user'] = 'test_user'
        df['task'] = 'test_task'
        # fill proxy columns used downstream
        for col, default in [('algo','test'), ('accuracy', 0.5), ('autoCompleteUsed', False),
                             ('rankOfPredictionUsed', -1), ('difficulty','medium'),
                             ('supported', False), ('training', False), ('isGroundTruth', False)]:
            if col not in df.columns:
                df[col] = default
        df['selectedIndices'] = '[]'
        df['selectedCoords'] = '[]'
        # ensure x,y exist
        if 'x' not in df.columns and '0' in df.columns and '1' in df.columns:
            df['x'] = df['0']
            df['y'] = df['1']
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def parse_selected_data(self, df):
        """Parse selectedIndices and selectedCoords columns"""
        print("Parsing selected indices and coordinates...")
        
        def safe_parse_indices(indices_str):
            try:
                if pd.isna(indices_str) or indices_str == '[]':
                    return []
                return ast.literal_eval(indices_str)
            except:
                return []
        
        def safe_parse_coords(coords_str):
            try:
                if pd.isna(coords_str) or coords_str == '[]':
                    return []
                # Parse the coordinate string format
                coords_str = coords_str.replace('\n', ' ').replace('[', '').replace(']', '')
                parts = coords_str.split()
                coords = []
                for i in range(0, len(parts), 3):
                    if i + 2 < len(parts):
                        try:
                            x = float(parts[i])
                            y = float(parts[i+1])
                            coords.append([x, y])
                        except:
                            continue
                return coords
            except:
                return []
        
        df['parsed_indices'] = df['selectedIndices'].apply(safe_parse_indices)
        df['parsed_coords'] = df['selectedCoords'].apply(safe_parse_coords)
        df['num_selected'] = df['parsed_indices'].apply(len)
        
        print(f"✅ Parsed selection data")
        return df
    
    def create_interaction_sequences(self, df):
        """Create interaction sequences from the track stories data"""
        print("Creating interaction sequences...")
        
        sequences = []
        labels = []
        
        # Group by user and line to create sequences
        for (user, line), group in df.groupby(['user', 'line']):
            if len(group) < 2:  # Skip single-point sequences
                continue
            
            # Sort by row order (assuming it represents time)
            group = group.sort_index()
            
            sequence = []
            prev_indices = []
            prev_coords = []
            
            for idx, row in group.iterrows():
                # Calculate spatial position
                spatial_position = [row['x'], row['y']]
                
                # Determine action type based on selection changes
                current_indices = row['parsed_indices']
                current_coords = row['parsed_coords']
                
                if len(current_indices) > len(prev_indices):
                    action_type = 'select'
                elif len(current_indices) < len(prev_indices):
                    action_type = 'deselect'
                elif len(current_indices) > 0 and len(prev_indices) > 0:
                    action_type = 'modify_selection'
                elif row['autoCompleteUsed']:
                    action_type = 'assist'
                else:
                    action_type = 'navigate'
                
                # Calculate interaction intensity
                if len(prev_coords) > 0 and len(current_coords) > 0:
                    # Calculate movement distance
                    prev_center = np.mean(prev_coords, axis=0) if prev_coords else [0, 0]
                    curr_center = np.mean(current_coords, axis=0) if current_coords else [0, 0]
                    movement = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
                else:
                    movement = 0
                
                interaction_intensity = min(movement / 10.0, 1.0)  # Normalize
                
                # Calculate data density (selection size)
                data_density = min(len(current_indices) / 100.0, 1.0)  # Normalize
                
                # Create interaction object
                interaction = {
                    'timestamp': idx,  # Use row index as timestamp
                    'action_type': action_type,
                    'interaction_intensity': interaction_intensity,
                    'spatial_position': spatial_position,
                    'data_density': data_density,
                    'data_dimension': len(current_indices),
                    'accuracy': row['accuracy'],
                    'difficulty': row['difficulty'],
                    'task': row['task'],
                    'user_id': user,
                    'line_id': line
                }
                
                sequence.append(interaction)
                prev_indices = current_indices
                prev_coords = current_coords
            
            if len(sequence) > 1:  # Only keep sequences with multiple interactions
                sequences.append(sequence)
                
                # Determine behavior type based on user patterns
                behavior_type = self._infer_behavior_type(sequence, row)
                labels.append(behavior_type)
        
        print(f"✅ Created {len(sequences)} interaction sequences")
        print(f"Average sequence length: {np.mean([len(seq) for seq in sequences]):.1f}")
        
        return sequences, labels
    
    def _infer_behavior_type(self, sequence, row):
        """Infer behavior type from sequence characteristics"""
        # Analyze sequence patterns
        actions = [interaction['action_type'] for interaction in sequence]
        intensities = [interaction['interaction_intensity'] for interaction in sequence]
        accuracies = [interaction['accuracy'] for interaction in sequence]
        
        # Calculate metrics
        action_diversity = len(set(actions)) / len(actions)
        avg_intensity = np.mean(intensities)
        avg_accuracy = np.mean(accuracies)
        difficulty = row['difficulty']
        
        # Classify based on patterns
        if action_diversity > 0.7 and avg_intensity > 0.5:
            return 'explorer'
        elif avg_accuracy > 0.7 and difficulty == 'hard':
            return 'analyst'
        elif action_diversity < 0.3 and avg_intensity < 0.3:
            return 'focused'
        else:
            return 'systematic'
    
    def load_real_data(self, dataset_name='track_stories'):
        """Main function to load real data"""
        print("Loading real data from projection-space-explorer...")
        
        # Download data
        filename = self.download_csv_data(dataset_name)
        if not filename:
            return None, None
        
        # Load and process data
        if dataset_name == 'track_stories':
            df = self.load_track_stories_data(filename)
        elif dataset_name == 'test_cluster':
            df = self.load_testcluster_data(filename)
        else:
            # default loader
            df = self.load_track_stories_data(filename)
        df = self.parse_selected_data(df)
        
        # Create sequences
        sequences, labels = self.create_interaction_sequences(df)
        
        print(f"\n📊 Real Data Summary:")
        print(f"  • Sequences: {len(sequences)}")
        print(f"  • Behavior types: {set(labels)}")
        print(f"  • Average length: {np.mean([len(seq) for seq in sequences]):.1f}")
        
        return sequences, labels

class RealDataProcessor:
    """Process real data for model training"""
    
    def __init__(self):
        self.data_loader = ProjectionSpaceDataLoader()
        self.scaler = StandardScaler()
        self.label_encoder = None
        
    def load_and_prepare_data(self, dataset_name='track_stories'):
        """Load real data and prepare for training"""
        print("Loading real projection-space-explorer data...")
        
        # Load real data
        sequences, labels = self.data_loader.load_real_data(dataset_name)
        
        if sequences is None:
            print("❌ Failed to load real data, falling back to synthetic data")
            from predictive_provenance_model import ProvenanceDataProcessor
            processor = ProvenanceDataProcessor()
            sequences, labels = processor.load_synthetic_data(n_samples=500)
        
        # Convert to feature matrix
        features = self._extract_features_from_real_data(sequences)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        return features_scaled, labels_encoded, sequences, labels
    
    def _extract_features_from_real_data(self, sequences):
        """Extract features from real interaction sequences"""
        features = []
        
        for seq in sequences:
            if len(seq) == 0:
                continue
            
            # Extract temporal features
            timestamps = [interaction['timestamp'] for interaction in seq]
            duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
            interaction_rate = len(seq) / max(duration, 1)
            
            # Extract action features
            actions = [interaction['action_type'] for interaction in seq]
            unique_actions = len(set(actions))
            action_diversity = unique_actions / len(actions) if len(actions) > 0 else 0
            
            # Extract spatial features
            positions = np.array([interaction['spatial_position'] for interaction in seq])
            spatial_variance = np.var(positions, axis=0).mean() if len(positions) > 1 else 0
            spatial_coverage = np.prod(np.max(positions, axis=0) - np.min(positions, axis=0))
            
            # Extract intensity features
            intensities = [interaction['interaction_intensity'] for interaction in seq]
            avg_intensity = np.mean(intensities)
            intensity_variance = np.var(intensities)
            
            # Extract accuracy features
            accuracies = [interaction['accuracy'] for interaction in seq]
            avg_accuracy = np.mean(accuracies)
            accuracy_variance = np.var(accuracies)
            
            # Extract data dimension features
            dimensions = [interaction['data_dimension'] for interaction in seq]
            dimension_diversity = len(set(dimensions)) / len(dimensions) if len(dimensions) > 0 else 0
            
            # Extract density features
            densities = [interaction['data_density'] for interaction in seq]
            avg_density = np.mean(densities)
            density_variance = np.var(densities)
            
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
                avg_accuracy,  # average accuracy
                accuracy_variance,  # accuracy variance
                dimension_diversity,  # dimension diversity
                avg_density,  # average data density
                density_variance,  # density variance
            ]
            
            # Add action type frequencies
            action_freq = {}
            for action in ['select', 'deselect', 'modify_selection', 'assist', 'navigate']:
                action_freq[action] = actions.count(action) / len(actions)
                feature_vector.append(action_freq[action])
            
            features.append(feature_vector)
        
        return np.array(features)

def main():
    """Test the real data loader"""
    print("Testing Real Data Loader")
    print("=" * 40)
    
    processor = RealDataProcessor()
    
    # Load real data
    X, y, sequences, labels = processor.load_and_prepare_data('track_stories')
    
    print(f"\nResults:")
    print(f"  • Feature matrix shape: {X.shape}")
    print(f"  • Sequences: {len(sequences)}")
    print(f"  • Behavior types: {set(labels)}")
    print(f"  • Average sequence length: {np.mean([len(seq) for seq in sequences]):.1f}")
    
    # Show sample sequence
    if sequences:
        print(f"\nSample sequence (first 3 interactions):")
        sample_seq = sequences[0][:3]
        for i, interaction in enumerate(sample_seq):
            print(f"  {i+1}. {interaction['action_type']} at position {interaction['spatial_position']}")
    
    print("\n✅ Real data loader test completed!")

if __name__ == "__main__":
    main()
