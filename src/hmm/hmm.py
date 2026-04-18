import numpy as np
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _get_device()


def belief_to_hex(belief):
    """Convert belief state probabilities to hex color codes."""
    if hasattr(belief, "detach"):
        belief = belief.detach().cpu().numpy()
    return [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in belief
    ]


def belief_to_barycentric(probs, belief_states):
    """Plot belief states on a barycentric (simplex) coordinate system."""
    sqrt3 = torch.sqrt(torch.tensor(3.0))
    x_coords = probs[:, 1] + 0.5 * probs[:, 2]
    y_coords = (sqrt3 / 2) * probs[:, 2]

    if not isinstance(x_coords, np.ndarray):
        df = pd.DataFrame({
            'x': x_coords.numpy(),
            'y': y_coords.numpy(),
            'p0': probs[:, 0].numpy(),
            'p1': probs[:, 1].numpy(),
            'p2': probs[:, 2].numpy()
        })
    else:
        df = pd.DataFrame({
            'x': x_coords,
            'y': y_coords,
            'p0': probs[:, 0],
            'p1': probs[:, 1],
            'p2': probs[:, 2]
        })

    df['color'] = belief_to_hex(belief_states)

    fig = px.scatter(
        df, x='x', y='y',
        title='Mixed-State Presentation (MSP) of Belief States with Color',
        labels={'x': 'Barycentric x-coordinate', 'y': 'Barycentric y-coordinate'},
        opacity=0.5
    )

    fig.update_traces(marker=dict(color=df['color'], size=2))

    S0 = (0, 0)
    S1 = (1, 0)
    SR = (0.5, (sqrt3 / 2).item())

    fig.add_trace(go.Scatter(
        x=[S0[0]], y=[S0[1]], mode='markers+text',
        marker=dict(color='blue', size=10),
        text=["S0"], textposition="bottom center"
    ))
    fig.add_trace(go.Scatter(
        x=[S1[0]], y=[S1[1]], mode='markers+text',
        marker=dict(color='blue', size=10),
        text=["S1"], textposition="bottom center"
    ))
    fig.add_trace(go.Scatter(
        x=[SR[0]], y=[SR[1]], mode='markers+text',
        marker=dict(color='blue', size=10),
        text=["SR"], textposition="top center"
    ))

    fig.update_xaxes(range=[-0.1, 1.1])
    fig.update_yaxes(range=[-0.1, (sqrt3 / 2).item() + 0.1], scaleanchor="x", scaleratio=1)

    fig.show()


def belief_to_barycentric_evolution(probs, belief_states):
    """Trace the movement of data point updates on the simplex."""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(belief_states, torch.Tensor):
        belief_states = belief_states.cpu().numpy()

    N, M, _ = belief_states.shape
    if probs.ndim == 2:
        probs = np.repeat(probs[:, None, :], M, axis=1)

    flat = probs.reshape(-1, 3)
    idx = np.tile(np.arange(M), N)

    sqrt3 = np.sqrt(3.0)
    x = flat[:, 1] + 0.5 * flat[:, 2]
    y = (sqrt3 / 2) * flat[:, 2]

    df = pd.DataFrame({
        'x': x,
        'y': y,
        'series_norm': idx / (M - 1),
    })
    fig = px.scatter(
        df, x='x', y='y',
        color='series_norm',
        color_continuous_scale='Viridis',
        range_color=(0, 1),
        labels={'x': 'Barycentric x', 'y': 'Barycentric y', 'series_norm': 'State idx (norm)'},
        opacity=0.6,
        title=f'MSP of Belief States — {N}×{M} points'
    )
    fig.update_traces(marker=dict(size=3))

    # Add one line-trace per step k→k+1
    for k in range(M - 1):
        k_norm = k / (M - 1)
        col = sample_colorscale('Viridis', [k_norm])[0]
        xs, ys = [], []

        # Collect N little segments
        for n in range(N):
            base = n * M
            i0, i1 = base + k, base + k + 1
            xs += [x[i0], x[i1], None]
            ys += [y[i0], y[i1], None]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=col, width=1),
            opacity=1,
            showlegend=False
        ))

    corners = [
        (0, 0, 'S0', 'bottom center'),
        (1, 0, 'S1', 'bottom center'),
        (0.5, sqrt3 / 2, 'SR', 'top center')
    ]
    for xx, yy, lbl, pos in corners:
        fig.add_trace(go.Scatter(
            x=[xx], y=[yy],
            mode='markers+text',
            marker=dict(color='blue', size=10),
            text=[lbl],
            textposition=pos
        ))

    fig.update_xaxes(range=[-0.1, 1.1])
    fig.update_yaxes(range=[-0.1, sqrt3 / 2 + 0.1], scaleanchor="x", scaleratio=1)

    fig.show()


class BeliefHMM:
    """Base class for HMMs that support belief-state computation.

    Subclasses must set:
        self.num_states      : int
        self.T_3d_matrix     : torch.Tensor (n_tokens, n_states, n_states)
                               column convention — T[v, next_state, cur_state]
        self.reverse_mapping : dict {vocab_idx -> internal_idx}
    """

    def generate_dataset(self, num_sequences, seq_length, return_states=False):
        """Generate sequences from the HMM.

        Returns:
            tokens, tokens_y if return_states=False
            tokens, tokens_y, states if return_states=True
        """
        vocab_size, n_states, _ = self.T_3d_matrix.shape
        prior = torch.ones(n_states, dtype=torch.float, device=device) / n_states

        tokens = torch.zeros(num_sequences, seq_length, dtype=torch.int64, device=device)
        tokens_y = torch.zeros(num_sequences, seq_length, dtype=torch.int64, device=device)
        if return_states:
            states = torch.zeros(num_sequences, seq_length, dtype=torch.int64, device=device)
        state_idx = torch.multinomial(prior, num_sequences, replacement=True)

        for i in range(seq_length + 1):
            if return_states and i < seq_length:
                states[:, i] = state_idx

            test = self.T_3d_matrix[:, :, state_idx].T.unsqueeze(0).reshape(num_sequences, -1)
            test_pairs = torch.multinomial(test, 1, replacement=True).squeeze()
            state_idx = test_pairs // vocab_size
            if i != seq_length:
                tokens[:, i] = test_pairs % vocab_size
            if i != 0:
                tokens_y[:, i - 1] = test_pairs % vocab_size

        if return_states:
            return tokens, tokens_y, states
        return tokens, tokens_y, None

    def compute_belief_state(self, tokens, initial_belief=None):
        """Compute belief states given a sequence of tokens."""
        if initial_belief is None:
            belief = torch.ones(self.num_states, device=device) / self.num_states
        else:
            belief = torch.tensor(initial_belief, dtype=torch.float, device=device)

        num_sequences, length = tokens.shape
        beliefs = torch.zeros(num_sequences, length + 1, self.num_states, dtype=torch.float, device=device)
        beliefs[:, 0] = belief

        internal_tokens = torch.zeros_like(tokens)
        for vocab_idx, internal_idx in self.reverse_mapping.items():
            internal_tokens[tokens == vocab_idx] = internal_idx

        for i in range(length):
            beliefs[:, i + 1] = (self.T_3d_matrix[internal_tokens[:, i]] @ beliefs[:, i, :, None]).squeeze()
            beliefs[:, i + 1] = beliefs[:, i + 1] / beliefs[:, i + 1].sum(-1, keepdim=True)

        return beliefs

    def emission_matrix(self):
        """Emission matrix (n_states, vocab_size): P(token | state)."""
        em = self.T_3d_matrix.sum(dim=1).T  # (n_states, vocab_size)  — sum over next_state (dim 1)
        return em / em.sum(dim=1, keepdim=True)


class FernHMM(BeliefHMM):
    """HMM for the Fern process (2 tokens, 3 states).

    Matrix from Xavier's xavier/processes branch, converted from row convention
    (belief @ T[tok]) to our column convention (T[tok] @ belief) via transpose(-1, -2).
    """

    def __init__(self, vocab_mapping=None):
        self.tokens = ['A', 'B']
        self.states = ['1', '2', '3']
        self.num_states = len(self.states)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}

        if vocab_mapping is None:
            self.vocab_mapping = {0: 0, 1: 1}
        else:
            self.vocab_mapping = vocab_mapping
        self.reverse_mapping = {v: k for k, v in self.vocab_mapping.items()}

        self.T_3d_matrix = torch.zeros((2, 3, 3), device=device)

    def create_hmm(self, x: float):
        """Build the Fern transition tensor for parameter x.

        Xavier's row-convention matrix is transposed to our column convention:
        T_col[v, next_state, cur_state] = T_xavier[v, cur_state, next_state]
        """
        T_xavier = torch.tensor([
            [[0.3942,       0.00512,        0.0381],
             [0.0,          0.53,           0.0],
             [0.0,          0.326 * x,      0.554]],
            [[0.3358,       0.01088,        0.2159],
             [0.0,          0.0,            0.47],
             [0.12,         0.326 * (1-x),  0.0]],
        ], dtype=torch.float32, device=device)
        self.T_3d_matrix = T_xavier.transpose(-1, -2)
        return self.T_3d_matrix


class Mess3HMM(BeliefHMM):
    """Hidden Markov Model for the Mess3 process with 3 states and 3 tokens."""

    def __init__(self, vocab_mapping=None, seed=42):
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        self.tokens = ['A', 'B', 'C']
        self.states = ['1', '2', '3']
        self.num_states = len(self.states)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.states_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_states = {idx: state for idx, state in enumerate(self.states)}

        self.transition_matrix = {
            'A': torch.tensor([
                [0.765, 0.00375, 0.00375],
                [0.0425, 0.0675, 0.00375],
                [0.0425, 0.00375, 0.0675]
            ], device=device),
            'B': torch.tensor([
                [0.0675, 0.0425, 0.00375],
                [0.00375, 0.765, 0.00375],
                [0.00375, 0.0425, 0.0675]
            ], device=device),
            'C': torch.tensor([
                [0.0675, 0.00375, 0.0425],
                [0.00375, 0.0675, 0.0425],
                [0.00375, 0.00375, 0.765]
            ], device=device)
        }

        self.T_3d_matrix = torch.stack([
            self.transition_matrix['A'],
            self.transition_matrix['B'],
            self.transition_matrix['C']
        ])

        if vocab_mapping is None:
            self.vocab_mapping = {0: 0, 1: 1, 2: 2}
        else:
            self.vocab_mapping = vocab_mapping

        self.reverse_mapping = {v: k for k, v in self.vocab_mapping.items()}

    def create_hmm(self, x, alpha):
        """Create HMM transition matrices with given parameters (from epsilon transformers)."""
        T = torch.zeros((3, 3, 3), device=device)
        beta = (1 - alpha) / 2
        y = 1 - 2 * x

        ay = alpha * y
        bx = beta * x
        by = beta * y
        ax = alpha * x

        T[0, :, :] = torch.tensor([[ay, bx, bx], [ax, by, bx], [ax, bx, by]], device=device)
        T[1, :, :] = torch.tensor([[by, ax, bx], [bx, ay, bx], [bx, ax, by]], device=device)
        T[2, :, :] = torch.tensor([[by, bx, ax], [bx, by, ax], [bx, bx, ay]], device=device)

        #x0.05, alpha0.9 => beta = 0.05, y = 0.9, ay=0.81 bx=0.025 ax=0.045
        # print(T)

        self.T_3d_matrix = T
        return T

    def sample_continuation(
        self,
        initial_belief: np.ndarray,
        seq_length: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        belief = np.asarray(initial_belief, dtype=np.float32).copy()
        emission = self.emission_matrix().detach().cpu().numpy()
        transition = self.T_3d_matrix.detach().cpu().numpy()
        tokens = np.zeros(seq_length, dtype=np.int64)
        beliefs = np.zeros((seq_length, self.num_states), dtype=np.float32)

        for idx in range(seq_length):
            probs = belief @ emission
            probs = probs / (probs.sum() + 1e-10)
            token_idx = int(rng.choice(emission.shape[1], p=probs))
            belief = transition[token_idx] @ belief
            belief = (belief / (belief.sum() + 1e-10)).astype(np.float32)
            tokens[idx] = token_idx
            beliefs[idx] = belief

        return tokens, beliefs

    def observation_probability_distribution(self, belief):
        """P(symbol | belief) for each symbol. belief: (n_states,) or (batch, n_states)."""
        em = self.emission_matrix()
        if belief.dim() == 1:
            return (belief.unsqueeze(0) @ em).squeeze(0)
        return belief @ em

    def log_probability(self, seq):
        """Log P(sequence) under the HMM. seq: 1D tensor of token indices (length L)."""
        if seq.dim() == 2:
            seq = seq.squeeze(0)
        seq = seq.to(device)
        belief = torch.ones(self.num_states, device=device) / self.num_states
        em = self.emission_matrix()
        log_p = 0.0
        for t in range(seq.size(0)):
            obs_probs = belief @ em  # (vocab_size,)
            log_p = log_p + torch.log(obs_probs[seq[t]] + 1e-10)
            # Update belief given observation
            belief = (self.T_3d_matrix[seq[t]] @ belief.unsqueeze(-1)).squeeze(-1)
            belief = belief / belief.sum()
        return log_p
