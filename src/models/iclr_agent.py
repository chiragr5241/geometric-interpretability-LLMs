import numpy as np
import networkx as nx
from .llm_interface import LLMInterface

class ICLRAgent:
    def __init__(self, 
                 embedding_dim=1536, 
                 hidden_dim=512):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize parameters
        self.W = np.random.randn(embedding_dim, hidden_dim) / np.sqrt(embedding_dim)
        self.b = np.zeros(hidden_dim)
        
        # Energy minimization parameters
        self.dirichlet_lambda = 0.1
        self.semantic_prior_weight = 0.5
        
        # Initialize LLM interface 
        self.llm = LLMInterface()
        
    def compute_dirichlet_energy(self, representations, graph):
        """Compute Dirichlet energy for given representations and graph structure."""
        energy = 0.0
        for i, j in graph.edges():
            diff = representations[i] - representations[j]
            energy += np.sum(diff * diff)
        return energy * self.dirichlet_lambda
    
    def compute_pca(self, data, n_components=2):
        """Simple PCA implementation using numpy."""
        # Center the data
        centered_data = data - np.mean(data, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.dot(centered_data.T, centered_data) / (data.shape[0] - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project data onto principal components
        return np.dot(centered_data, eigenvectors[:, :n_components])
    
    def reorganize_representations(self, context, graph):
        """Reorganize semantic representations based on context and graph structure."""
        # Generate embeddings using OpenAI for context items
        if isinstance(context[0], str):
            embeddings = np.array([self.llm.get_embedding(str(item)) for item in context])
        else:
            embeddings = np.random.randn(len(context), self.embedding_dim)
        
        # Simple linear transformation to project the embeddings into a lower-dimensional space 512
        hidden = np.tanh(np.dot(embeddings, self.W) + self.b)
        
        # Compute Dirichlet energy
        energy = self.compute_dirichlet_energy(hidden, graph)
        
        # Apply simple gradient step to minimize energy
        step_size = 0.01
        grad = np.zeros_like(hidden)
        for i, j in graph.edges():
            diff = hidden[i] - hidden[j]
            grad[i] += diff
            grad[j] -= diff
        
        reorganized = hidden - step_size * grad
        
        # Get semantic analysis of the reorganization
        analysis = self.llm.generate_semantic_description(
            reorganized, 
            graph.__class__.__name__, 
            len(context)
        )
        
        return reorganized, energy, analysis
    
    def trace_graph(self, graph_type='grid', context_size=10):
        """Execute graph tracing task with specified parameters."""
        # Create graph based on type
        if graph_type == 'grid':
            n = int(np.sqrt(context_size))
            graph = nx.grid_2d_graph(n, n)
            # Convert 2D coordinates to indices
            mapping = {(x, y): y * n + x for x, y in graph.nodes()}
            graph = nx.relabel_nodes(graph, mapping)
            # print(mapping)
            # print(graph.nodes())
            # print(graph.edges())
        elif graph_type == 'ring':
            graph = nx.cycle_graph(context_size)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")
            
        # Generate context (using numbers as strings for semantic meaning)
        context = [str(i) for i in range(context_size)]

        # Process context and reorganize representations
        # representations, energy, analysis = self.reorganize_representations(context, graph)
        
        # # Compute PCA for visualization
        # pca_result = self.compute_pca(representations, n_components=2)
        
        return {
            # 'representations': representations,
            # 'energy': energy,
            # 'pca_visualization': pca_result,
            'graph': graph,
            # 'semantic_analysis': analysis
        }
    
    def analyze_semantic_priors(self, prior_type='weekdays'):
        """Analyze influence of semantic priors on representations."""
        if prior_type == 'weekdays':
            priors = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            graph = nx.cycle_graph(len(priors))
            
            # Get embeddings for weekdays using OpenAI
            embeddings = np.array([self.llm.get_embedding(day) for day in priors])
            
            # Get analysis of the semantic structure
            analysis = self.llm.analyze_representation(
                embeddings.mean(axis=0),
                "Analyzing weekday semantic structure in cyclic graph"
            )
            
            return {
                'embeddings': embeddings,
                'graph': graph,
                'labels': priors,
                'analysis': analysis
            }
        
        return None
