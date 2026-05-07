import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx

class PerformanceMetrics:
    def __init__(self):
        self.metrics_history = {}
        
    def compute_representation_alignment(self, representations, graph):
        """Compute alignment between representations and graph structure."""
        # Compute pairwise distances in representation space
        rep_distances = pairwise_distances(representations)
        
        # Compute graph distances
        graph_distances = nx.floyd_warshall_numpy(graph)
        
        # Compute correlation between distances
        correlation = np.corrcoef(rep_distances.flatten(), 
                                graph_distances.flatten())[0, 1]
        
        return correlation
    
    def compute_dirichlet_quotient(self, representations, graph):
        """Compute Dirichlet quotient as measure of representation quality."""
        numerator = 0
        denominator = 0
        
        for i, j in graph.edges():
            diff = representations[i] - representations[j]
            numerator += np.sum(diff * diff)
            
        for i in range(len(representations)):
            for j in range(len(representations)):
                if i != j:
                    diff = representations[i] - representations[j]
                    denominator += np.sum(diff * diff)
                    
        return numerator / denominator if denominator != 0 else float('inf')
    
    def evaluate_semantic_preservation(self, original_semantics, new_representations):
        """Evaluate how well original semantic relationships are preserved."""
        original_distances = pairwise_distances(original_semantics)
        new_distances = pairwise_distances(new_representations)
        
        preservation_score = np.corrcoef(original_distances.flatten(),
                                       new_distances.flatten())[0, 1]
        
        return preservation_score
    
    def compute_phase_transition_metrics(self, accuracies):
        """Compute metrics related to phase transition behavior."""
        # Detect phase transition point
        accuracy_diff = np.diff(accuracies)
        transition_point = np.argmax(accuracy_diff)
        
        # Compute transition sharpness
        transition_sharpness = accuracy_diff[transition_point]
        
        return {
            'transition_point': transition_point,
            'transition_sharpness': transition_sharpness,
            'pre_transition_mean': np.mean(accuracies[:transition_point]),
            'post_transition_mean': np.mean(accuracies[transition_point+1:])
        }
    
    def update_metrics_history(self, metric_name, value):
        """Update metrics history for tracking performance over time."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        self.metrics_history[metric_name].append(value)
        
    def get_summary_statistics(self):
        """Get summary statistics for all tracked metrics."""
        summary = {}
        for metric_name, values in self.metrics_history.items():
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return summary
