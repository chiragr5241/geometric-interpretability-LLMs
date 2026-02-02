import os
from dotenv import load_dotenv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaInterface:
    """Interface for Llama models to extract layer-wise hidden states."""
    
    SUPPORTED_MODELS = {
        "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
        "gemma2-2b": "google/gemma-2-2b",
        "gemma2-9b": "google/gemma-2-9b",
    }
    
    def __init__(self, model_name="llama-3.2-1b", device=None):
        """
        Initialize Llama model.
        
        Args:
            model_name: One of "llama-3.1-8b" or "llama-3.2-1b"
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        load_dotenv()
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS[model_name]
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=os.getenv('HF_TOKEN')
        )
        
        # Load model with appropriate settings
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                token=os.getenv('HF_TOKEN')
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                token=os.getenv('HF_TOKEN')
            ).to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully! Layers: {self.model.config.num_hidden_layers}")
        
    @property
    def num_layers(self):
        """Return number of hidden layers in the model."""
        return self.model.config.num_hidden_layers
    
    @property
    def hidden_size(self):
        """Return hidden dimension of the model."""
        return self.model.config.hidden_size
        
    def tokenize(self, text):
        """Tokenize text and return token ids and tokens."""
        encoded = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(encoded.input_ids[0])
        return encoded.input_ids[0].tolist(), tokens
    
    def get_hidden_states(self, text, return_tokens=False):
        """
        Get hidden states from all layers for the input text.
        
        Args:
            text: Input text string
            return_tokens: Whether to also return the tokens
            
        Returns:
            hidden_states: numpy array of shape (num_layers+1, seq_len, hidden_dim)
                          Layer 0 is the embedding layer, layers 1-N are transformer layers
            tokens: (optional) list of token strings
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract hidden states: tuple of (num_layers+1) tensors of shape (batch, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states

        print(f"Hidden states: {len(hidden_states)} layers, each shape: {hidden_states[0].shape}")
        
        # Stack and convert to numpy: (num_layers+1, seq_len, hidden_dim)
        hidden_states_np = np.stack([
            h.squeeze(0).cpu().numpy() for h in hidden_states
        ])
        
        if return_tokens:
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            return hidden_states_np, tokens
        
        return hidden_states_np
    
    def get_layer_representations(self, text, layer_idx=-1):
        """
        Get representations from a specific layer.
        
        Args:
            text: Input text string
            layer_idx: Layer index (0=embedding, 1-N=transformer layers, -1=last layer)
            
        Returns:
            representations: numpy array of shape (seq_len, hidden_dim)
        """
        hidden_states = self.get_hidden_states(text)
        return hidden_states[layer_idx]
    
    def get_token_representations(self, text, token_indices, layer_idx=-1):
        """
        Get representations for specific tokens at a specific layer.
        
        Args:
            text: Input text string
            token_indices: List of token indices to extract
            layer_idx: Layer index
            
        Returns:
            representations: numpy array of shape (len(token_indices), hidden_dim)
        """
        hidden_states = self.get_hidden_states(text)
        return hidden_states[layer_idx][token_indices]
    
    def get_mean_token_representations(self, texts, layer_idx=-1):
        """
        Get mean representation for each text across all its tokens.
        
        Args:
            texts: List of text strings
            layer_idx: Layer index
            
        Returns:
            representations: numpy array of shape (len(texts), hidden_dim)
        """
        representations = []
        for text in texts:
            hidden = self.get_layer_representations(text, layer_idx)
            representations.append(hidden.mean(axis=0))
        return np.array(representations)
    
    def forward_with_context(self, context_sequence, layer_idx=26):
        """
        Process a context sequence (like random walk on graph) and extract representations.
        This is the key method for replicating the paper's experiments.
        
        Args:
            context_sequence: List of tokens/words forming the random walk sequence
            layer_idx: Which layer to extract representations from (paper uses layer 26 for 8B)
            
        Returns:
            representations: dict with token representations from the specified layer
        """
        # Join tokens into a sequence
        if isinstance(context_sequence, list):
            text = " ".join(context_sequence)
        else:
            text = context_sequence
        
        # Get hidden states and tokens
        hidden_states, tokens = self.get_hidden_states(text, return_tokens=True)
        
        # Adjust layer index for smaller models
        actual_layer = min(layer_idx, self.num_layers)
        
        return {
            'hidden_states': hidden_states,
            'layer_representations': hidden_states[actual_layer],
            'tokens': tokens,
            'layer_idx': actual_layer,
            'all_layers': self.num_layers
        }
    
    def generate(self, prompt, max_new_tokens=50, temperature=0.7):
        """Generate text continuation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Keep OpenAI interface for backward compatibility
class OpenAIInterface:
    """OpenAI API interface for embeddings and analysis."""
    
    def __init__(self):
        load_dotenv()
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4-1106-preview"
        
    def get_embedding(self, text):
        """Get embedding for a given text using OpenAI's embedding model."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def analyze_representation(self, representation, context):
        """Analyze a representation using GPT-4."""
        rep_summary = {
            'dimensions': len(representation),
            'mean': float(np.mean(representation)),
            'std': float(np.std(representation)),
        }
        
        prompt = f"""Analyze this neural representation in the context of in-context learning:
Context: {context}
Representation Statistics: {rep_summary}

Please provide insights on:
1. How this representation aligns with the given context
2. What patterns or structures are evident
3. Potential implications for in-context learning

Analysis:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def generate_semantic_description(self, representations, graph_type, context_size):
        """Generate a semantic description of the learned representations."""
        stats = {
            'mean_norm': float(np.mean([np.linalg.norm(r) for r in representations])),
            'std_norm': float(np.std([np.linalg.norm(r) for r in representations])),
            'dimensionality': representations.shape[1],
            'num_points': len(representations)
        }
        
        prompt = f"""Analyze these learned representations from an in-context learning task:
Graph Type: {graph_type}
Context Size: {context_size}
Statistics: {stats}

Please describe:
1. The emergent structure in the representations
2. How the graph type might influence the organization
3. Any phase transition indicators at this context size

Analysis:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content


# Backward compatibility alias
LLMInterface = OpenAIInterface
