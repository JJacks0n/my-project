"""
Advanced Visualization Tools for Sensemaking Trajectories

This module provides comprehensive visualization capabilities for analyzing
and understanding user behavior patterns in sensemaking activities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import os
import warnings
warnings.filterwarnings('ignore')

class TrajectoryVisualizer:
    """Comprehensive trajectory visualization toolkit"""
    
    def __init__(self, save_dir='visualizations'):
        self.colors = {
            'explorer': '#FF6B6B',
            'analyst': '#4ECDC4', 
            'focused': '#45B7D1',
            'random': '#96CEB4',
            'systematic': '#FFEAA7',
            'creative': '#DDA0DD'
        }
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_trajectory_2d(self, sequences, labels, max_sequences=100, title="Sensemaking Trajectories"):
        """Plot 2D trajectory visualization"""
        fig = go.Figure()
        
        for i, (seq, label) in enumerate(zip(sequences[:max_sequences], labels[:max_sequences])):
            if len(seq) == 0:
                continue
                
            # Extract positions
            positions = np.array([interaction['spatial_position'] for interaction in seq])
            timestamps = [interaction['timestamp'] for interaction in seq]
            actions = [interaction['action_type'] for interaction in seq]
            
            # Create trajectory line
            fig.add_trace(go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='lines+markers',
                line=dict(color=self.colors.get(label, '#999999'), width=3),
                marker=dict(size=8, symbol='circle'),
                name=f'User {i} ({label})',
                showlegend=False,
                hovertemplate=f'User {i} ({label})<br>Step: %{{text}}<br>Action: %{{customdata}}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>',
                text=timestamps,
                customdata=actions
            ))
            
            # Add start and end markers
            fig.add_trace(go.Scatter(
                x=[positions[0, 0]], y=[positions[0, 1]],
                mode='markers',
                marker=dict(size=12, color='green', symbol='star'),
                name=f'Start {i}',
                showlegend=False,
                hovertemplate=f'Start: User {i}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=[positions[-1, 0]], y=[positions[-1, 1]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name=f'End {i}',
                showlegend=False,
                hovertemplate=f'End: User {i}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Spatial Position X',
            yaxis_title='Spatial Position Y',
            width=1000,
            height=700,
            showlegend=True
        )
        
        # Save plot
        fig.write_html(os.path.join(self.save_dir, 'trajectory_2d.html'))
        try:
            fig.write_image(os.path.join(self.save_dir, 'trajectory_2d.png'))
        except:
            print("Note: PNG export requires kaleido. HTML saved instead.")
        
        return fig
    
    def plot_behavior_heatmap(self, sequences, labels):
        """Create behavior pattern heatmap"""
        # Extract features
        features = []
        for seq in sequences:
            if len(seq) == 0:
                continue
                
            # Calculate behavior metrics
            actions = [interaction['action_type'] for interaction in seq]
            positions = np.array([interaction['spatial_position'] for interaction in seq])
            intensities = [interaction['interaction_intensity'] for interaction in seq]
            
            # Calculate metrics
            spatial_variance = np.var(positions, axis=0).mean() if len(positions) > 1 else 0
            action_diversity = len(set(actions)) / len(actions) if len(actions) > 0 else 0
            avg_intensity = np.mean(intensities)
            spatial_coverage = np.prod(np.max(positions, axis=0) - np.min(positions, axis=0))
            
            features.append([
                len(seq),  # sequence length
                spatial_variance,
                action_diversity,
                avg_intensity,
                spatial_coverage
            ])
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=[
            'Sequence Length', 'Spatial Variance', 'Action Diversity', 
            'Avg Intensity', 'Spatial Coverage'
        ])
        
        # Create correlation heatmap (only numeric features)
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Behavior Pattern Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'behavior_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show behavior type distribution separately
        plt.figure(figsize=(8, 6))
        behavior_counts = pd.Series(labels[:len(features)]).value_counts()
        plt.pie(behavior_counts.values, labels=behavior_counts.index, autopct='%1.1f%%')
        plt.title('Behavior Type Distribution')
        plt.savefig(os.path.join(self.save_dir, 'behavior_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def plot_action_sequence_network(self, sequences, labels, min_frequency=5):
        """Create network visualization of action sequences"""
        # Count action transitions
        transitions = {}
        for seq, label in zip(sequences, labels):
            actions = [interaction['action_type'] for interaction in seq]
            for i in range(len(actions) - 1):
                transition = f"{actions[i]} -> {actions[i+1]}"
                if transition not in transitions:
                    transitions[transition] = {'count': 0, 'labels': []}
                transitions[transition]['count'] += 1
                transitions[transition]['labels'].append(label)
        
        # Filter by minimum frequency
        filtered_transitions = {k: v for k, v in transitions.items() if v['count'] >= min_frequency}
        
        # Create network graph
        G = nx.DiGraph()
        
        for transition, data in filtered_transitions.items():
            source, target = transition.split(' -> ')
            G.add_edge(source, target, weight=data['count'], 
                      labels=data['labels'], count=data['count'])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create plotly network
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} → {edge[1]}<br>Frequency: {weight}")
        
        # Create edge traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Calculate node statistics
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            node_info.append(f"Node: {node}<br>In-degree: {in_degree}<br>Out-degree: {out_degree}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Action Sequence Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Network of action transitions in sensemaking trajectories",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="black", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        # Save plot
        fig.write_html(os.path.join(self.save_dir, 'action_network.html'))
        try:
            fig.write_image(os.path.join(self.save_dir, 'action_network.png'))
        except:
            print("Note: PNG export requires kaleido. HTML saved instead.")
        
        return fig
    
    def plot_temporal_analysis(self, sequences, labels):
        """Plot temporal analysis of trajectories"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sequence Length Distribution', 'Interaction Rate Over Time',
                          'Action Type Frequency', 'Spatial Movement Over Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Sequence length distribution
        lengths = [len(seq) for seq in sequences]
        fig.add_trace(
            go.Histogram(x=lengths, nbinsx=20, name='Sequence Length'),
            row=1, col=1
        )
        
        # 2. Interaction rate over time
        all_timestamps = []
        all_rates = []
        for seq in sequences:
            if len(seq) > 1:
                timestamps = [interaction['timestamp'] for interaction in seq]
                rates = []
                for i in range(1, len(timestamps)):
                    rate = 1 / (timestamps[i] - timestamps[i-1]) if timestamps[i] != timestamps[i-1] else 0
                    rates.append(rate)
                all_timestamps.extend(timestamps[1:])
                all_rates.extend(rates)
        
        if all_timestamps and all_rates:
            fig.add_trace(
                go.Scatter(x=all_timestamps, y=all_rates, mode='markers', name='Interaction Rate'),
                row=1, col=2
            )
        
        # 3. Action type frequency
        all_actions = []
        for seq in sequences:
            all_actions.extend([interaction['action_type'] for interaction in seq])
        
        action_counts = pd.Series(all_actions).value_counts()
        fig.add_trace(
            go.Bar(x=action_counts.index, y=action_counts.values, name='Action Frequency'),
            row=2, col=1
        )
        
        # 4. Spatial movement over time
        for i, (seq, label) in enumerate(zip(sequences[:10], labels[:10])):  # Limit to 10 sequences
            if len(seq) > 1:
                timestamps = [interaction['timestamp'] for interaction in seq]
                positions = np.array([interaction['spatial_position'] for interaction in seq])
                distances = []
                for j in range(1, len(positions)):
                    dist = np.linalg.norm(positions[j] - positions[j-1])
                    distances.append(dist)
                
                fig.add_trace(
                    go.Scatter(x=timestamps[1:], y=distances, mode='lines+markers',
                              name=f'User {i} ({label})', showlegend=True),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, title_text="Temporal Analysis of Sensemaking Trajectories")
        
        # Save plot
        fig.write_html(os.path.join(self.save_dir, 'temporal_analysis.html'))
        try:
            fig.write_image(os.path.join(self.save_dir, 'temporal_analysis.png'))
        except:
            print("Note: PNG export requires kaleido. HTML saved instead.")
        
        return fig
    
    def plot_behavior_clusters(self, sequences, labels):
        """Plot behavior clusters using dimensionality reduction"""
        # Extract features for clustering
        features = []
        for seq in sequences:
            if len(seq) == 0:
                continue
                
            actions = [interaction['action_type'] for interaction in seq]
            positions = np.array([interaction['spatial_position'] for interaction in seq])
            intensities = [interaction['interaction_intensity'] for interaction in seq]
            
            # Calculate features
            spatial_variance = np.var(positions, axis=0).mean() if len(positions) > 1 else 0
            action_diversity = len(set(actions)) / len(actions) if len(actions) > 0 else 0
            avg_intensity = np.mean(intensities)
            spatial_coverage = np.prod(np.max(positions, axis=0) - np.min(positions, axis=0))
            sequence_length = len(seq)
            
            features.append([
                sequence_length, spatial_variance, action_diversity, 
                avg_intensity, spatial_coverage
            ])
        
        features = np.array(features)
        valid_labels = labels[:len(features)]
        
        # Apply dimensionality reduction
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(features)
        
        # UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_result = umap_reducer.fit_transform(features)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('t-SNE', 'UMAP', 'PCA', 'Behavior Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot t-SNE
        for label in np.unique(valid_labels):
            mask = valid_labels == label
            fig.add_trace(
                go.Scatter(x=tsne_result[mask, 0], y=tsne_result[mask, 1],
                          mode='markers', name=label, marker=dict(color=self.colors.get(label, '#999999'))),
                row=1, col=1
            )
        
        # Plot UMAP
        for label in np.unique(valid_labels):
            mask = valid_labels == label
            fig.add_trace(
                go.Scatter(x=umap_result[mask, 0], y=umap_result[mask, 1],
                          mode='markers', name=label, marker=dict(color=self.colors.get(label, '#999999')),
                          showlegend=False),
                row=1, col=2
            )
        
        # Plot PCA
        for label in np.unique(valid_labels):
            mask = valid_labels == label
            fig.add_trace(
                go.Scatter(x=pca_result[mask, 0], y=pca_result[mask, 1],
                          mode='markers', name=label, marker=dict(color=self.colors.get(label, '#999999')),
                          showlegend=False),
                row=2, col=1
            )
        
        # Plot behavior distribution
        behavior_counts = pd.Series(valid_labels).value_counts()
        fig.add_trace(
            go.Bar(x=behavior_counts.index, y=behavior_counts.values, name='Behavior Count'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Behavior Clustering Analysis")
        
        # Save plot
        fig.write_html(os.path.join(self.save_dir, 'behavior_clusters.html'))
        try:
            fig.write_image(os.path.join(self.save_dir, 'behavior_clusters.png'))
        except:
            print("Note: PNG export requires kaleido. HTML saved instead.")
        
        return fig
    
    def create_interactive_dashboard(self, sequences, labels):
        """Create an interactive dashboard for trajectory analysis"""
        # Create multiple visualizations
        trajectory_fig = self.plot_trajectory_2d(sequences, labels, max_sequences=50)
        network_fig = self.plot_action_sequence_network(sequences, labels)
        temporal_fig = self.plot_temporal_analysis(sequences, labels)
        cluster_fig = self.plot_behavior_clusters(sequences, labels)
        
        # Create dashboard layout
        dashboard = go.Figure()
        
        # Add all traces to dashboard (simplified version)
        dashboard.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='Dashboard Placeholder'))
        
        dashboard.update_layout(
            title="Sensemaking Trajectory Analysis Dashboard",
            height=1000,
            showlegend=True
        )
        
        return {
            'trajectory': trajectory_fig,
            'network': network_fig,
            'temporal': temporal_fig,
            'clusters': cluster_fig,
            'dashboard': dashboard
        }

def main():
    """Demonstrate trajectory visualization capabilities"""
    print("Trajectory Visualization Toolkit")
    print("=" * 40)
    
    # Generate sample data
    from predictive_provenance_model import ProvenanceDataProcessor
    processor = ProvenanceDataProcessor()
    sequences, labels = processor.load_synthetic_data(n_samples=200)
    
    # Initialize visualizer
    visualizer = TrajectoryVisualizer()
    
    # Create visualizations
    print("Creating trajectory visualizations...")
    
    # 2D trajectory plot
    trajectory_fig = visualizer.plot_trajectory_2d(sequences, labels, max_sequences=30)
    trajectory_fig.show()
    
    # Behavior heatmap
    print("Creating behavior heatmap...")
    visualizer.plot_behavior_heatmap(sequences, labels)
    
    # Action sequence network
    print("Creating action sequence network...")
    network_fig = visualizer.plot_action_sequence_network(sequences, labels)
    network_fig.show()
    
    # Temporal analysis
    print("Creating temporal analysis...")
    temporal_fig = visualizer.plot_temporal_analysis(sequences, labels)
    temporal_fig.show()
    
    # Behavior clusters
    print("Creating behavior cluster analysis...")
    cluster_fig = visualizer.plot_behavior_clusters(sequences, labels)
    cluster_fig.show()
    
    # Interactive dashboard
    print("Creating interactive dashboard...")
    dashboard = visualizer.create_interactive_dashboard(sequences, labels)
    
    print("Visualization demonstration completed!")
    print("All plots have been generated and can be viewed interactively.")

if __name__ == "__main__":
    main()
