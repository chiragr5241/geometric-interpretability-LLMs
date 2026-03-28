"""
Manim animation explaining the Tuned Lens technique applied to Llama-3.2-3B.

Renders a mathematical + code-level walkthrough of:
  1. The problem: reading intermediate transformer layers
  2. Logit Lens baseline
  3. Tuned Lens: per-layer affine translators
  4. Training objective (KL divergence)
  5. Results on the Mess-3 HMM process

Run:
    manim -pql experiments/tuned_lens_manim.py TunedLensExplainer
    manim -pqh experiments/tuned_lens_manim.py TunedLensExplainer   # high quality
"""

from manim import *
import numpy as np

# ── Real experimental data ──────────────────────────────────────────────
LAYERS = list(range(28))
KL_FINAL_TUNED = [
    0.0116, 0.0104, 0.0094, 0.0094, 0.0088, 0.0086, 0.0090, 0.0095,
    0.0092, 0.0099, 0.0103, 0.0104, 0.0094, 0.0089, 0.0052, 0.0050,
    0.0039, 0.0031, 0.0028, 0.0027, 0.0024, 0.0015, 0.0015, 0.0016,
    0.0015, 0.0020, 0.0010, 0.0016,
]
KL_HMM_LOGIT = [
    None, None, 1.549, 0.914, 0.938, 0.664, 0.530, 0.584,
    0.533, 0.717, 0.773, 0.822, 0.494, 0.287, 0.161, 0.154,
    0.221, 0.142, 0.136, 0.117, 0.673, 0.645, 0.270, 0.152,
    0.167, 0.608, 0.572, 0.049,
]
KL_HMM_TUNED = [
    0.035, 0.038, 0.039, 0.039, 0.040, 0.040, 0.040, 0.041,
    0.040, 0.042, 0.042, 0.041, 0.043, 0.042, 0.043, 0.046,
    0.047, 0.048, 0.049, 0.050, 0.050, 0.049, 0.049, 0.049,
    0.048, 0.050, 0.048, 0.043,
]
R2_BELIEF = [
    0.993, 0.999, 0.999, 0.999, 0.999, 0.998, 0.998, 0.998,
    0.997, 0.995, 0.993, 0.992, 0.991, 0.991, 0.990, 0.991,
    0.989, 0.990, 0.991, 0.991, 0.990, 0.988, 0.989, 0.987,
    0.988, 0.987, 0.988, 0.985,
]
TOP1_TUNED = [
    0.970, 0.973, 0.971, 0.973, 0.973, 0.973, 0.973, 0.973,
    0.973, 0.970, 0.970, 0.969, 0.972, 0.971, 0.979, 0.979,
    0.983, 0.983, 0.983, 0.984, 0.983, 0.990, 0.988, 0.990,
    0.989, 0.989, 0.993, 0.992,
]


# ── Color palette ───────────────────────────────────────────────────────
BG = "#0f0f23"
ACCENT = "#4fc3f7"
GOLD = "#ffd54f"
GREEN = "#81c784"
RED = "#e57373"
PURPLE = "#ce93d8"
ORANGE = "#ffb74d"


class TunedLensExplainer(Scene):
    """Full walkthrough of the Tuned Lens technique."""

    def construct(self):
        self.camera.background_color = BG
        self.title_scene()
        self.problem_scene()
        self.logit_lens_scene()
        self.tuned_lens_math_scene()
        self.training_scene()
        self.code_walkthrough_scene()
        self.results_scene()
        self.key_insights_scene()

    # ── 1. Title ────────────────────────────────────────────────────────
    def title_scene(self):
        title = Text("Tuned Lens", font_size=72, color=ACCENT)
        subtitle = Text(
            "Reading Every Layer of Llama-3.2-3B",
            font_size=32, color=WHITE,
        ).next_to(title, DOWN, buff=0.5)
        ref = Text(
            "nostalgebraist-lol & belrose et al., arXiv:2303.08112",
            font_size=20, color=GREY,
        ).next_to(subtitle, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1)
        self.play(FadeIn(ref, shift=UP * 0.2), run_time=0.5)
        self.wait(2)
        self.play(FadeOut(VGroup(title, subtitle, ref)))

    # ── 2. The Problem ──────────────────────────────────────────────────
    def problem_scene(self):
        heading = Text("The Problem", font_size=48, color=GOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading), run_time=0.8)

        # Draw transformer as stacked blocks
        layers = VGroup()
        labels = VGroup()
        n_show = 8  # show 8 representative layers
        layer_indices = [0, 3, 7, 12, 17, 21, 25, 27]
        colors = [interpolate_color(ManimColor(RED), ManimColor(GREEN), i / 7) for i in range(8)]

        for i, (li, c) in enumerate(zip(layer_indices, colors)):
            rect = RoundedRectangle(
                width=3, height=0.5, corner_radius=0.1,
                fill_color=c, fill_opacity=0.4, stroke_color=c,
            ).shift(UP * (i * 0.7 - 2.4))
            lab = Text(f"Layer {li}", font_size=18, color=WHITE).move_to(rect)
            layers.add(rect)
            labels.add(lab)

        # Arrows between layers
        arrows = VGroup()
        for i in range(len(layers) - 1):
            arr = Arrow(
                layers[i].get_top(), layers[i + 1].get_bottom(),
                buff=0.05, stroke_width=2, color=GREY,
            )
            arrows.add(arr)

        transformer = VGroup(layers, labels, arrows).shift(LEFT * 3.5)
        self.play(
            LaggedStart(*[FadeIn(l, shift=UP * 0.2) for l in layers], lag_ratio=0.1),
            LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.1),
            run_time=2,
        )
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05), run_time=1)

        # Question mark at intermediate layer
        q_text = VGroup(
            Text("Each layer produces a", font_size=24, color=WHITE),
            MathTex(r"h_\ell \in \mathbb{R}^{3072}", font_size=36, color=ACCENT),
            Text("residual stream vector.", font_size=24, color=WHITE),
            Text("", font_size=12),
            Text("What does each layer", font_size=24, color=GOLD),
            Text('"know" about the next token?', font_size=24, color=GOLD),
        ).arrange(DOWN, buff=0.2).shift(RIGHT * 2.5)

        self.play(FadeIn(q_text, shift=LEFT * 0.3), run_time=1.5)
        self.wait(2.5)
        self.play(FadeOut(VGroup(transformer, heading, q_text)))

    # ── 3. Logit Lens ──────────────────────────────────────────────────
    def logit_lens_scene(self):
        heading = Text("Logit Lens (Baseline)", font_size=48, color=GOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading), run_time=0.8)

        # Pipeline diagram
        steps = [
            (r"h_\ell", "Residual\nstream", RED),
            (r"\text{LN}_\text{final}", "Layer\nnorm", ORANGE),
            (r"W_U \cdot (\,) + b_U", "Unembed", PURPLE),
            (r"\text{softmax}", "Softmax", GREEN),
            (r"p_\text{logit}", "Distribution", ACCENT),
        ]

        boxes = VGroup()
        math_labels = VGroup()
        desc_labels = VGroup()
        for i, (tex, desc, col) in enumerate(steps):
            box = RoundedRectangle(
                width=2.2, height=1.2, corner_radius=0.1,
                fill_color=col, fill_opacity=0.15, stroke_color=col,
            ).shift(RIGHT * (i * 2.8 - 5.6) + DOWN * 0.3)
            mt = MathTex(tex, font_size=28, color=col).move_to(box.get_center() + UP * 0.15)
            dt = Text(desc, font_size=16, color=GREY).move_to(box.get_center() + DOWN * 0.3)
            boxes.add(box)
            math_labels.add(mt)
            desc_labels.add(dt)

        pipe_arrows = VGroup()
        for i in range(len(boxes) - 1):
            arr = Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                buff=0.1, stroke_width=2, color=GREY,
            )
            pipe_arrows.add(arr)

        pipeline = VGroup(boxes, math_labels, desc_labels, pipe_arrows)

        self.play(
            LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.15),
            LaggedStart(*[Write(m) for m in math_labels], lag_ratio=0.15),
            LaggedStart(*[FadeIn(d) for d in desc_labels], lag_ratio=0.15),
            run_time=2,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in pipe_arrows], lag_ratio=0.1),
            run_time=1,
        )

        # Problem annotation
        problem = VGroup(
            Text("Problem:", font_size=28, color=RED),
            Text("Early layers haven't aligned representations", font_size=22, color=WHITE),
            Text("to the unembedding space yet!", font_size=22, color=WHITE),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.8)

        self.play(FadeIn(problem, shift=UP * 0.3), run_time=1)
        self.wait(2.5)
        self.play(FadeOut(VGroup(pipeline, heading, problem)))

    # ── 4. Tuned Lens Math ──────────────────────────────────────────────
    def tuned_lens_math_scene(self):
        heading = Text("Tuned Lens: The Idea", font_size=48, color=GOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading), run_time=0.8)

        # Core idea
        idea = VGroup(
            Text("Insert a learned affine probe at each layer:", font_size=26, color=WHITE),
        ).next_to(heading, DOWN, buff=0.5)
        self.play(FadeIn(idea), run_time=0.8)

        # Translator equation
        translator_eq = MathTex(
            r"T_\ell(h_\ell) = W_\ell \, h_\ell + b_\ell",
            font_size=48, color=ACCENT,
        ).next_to(idea, DOWN, buff=0.6)

        where = VGroup(
            MathTex(r"W_\ell \in \mathbb{R}^{d \times d}", font_size=30, color=WHITE),
            MathTex(r"b_\ell \in \mathbb{R}^{d}", font_size=30, color=WHITE),
            MathTex(r"d = 3072 \text{ (Llama-3.2-3B)}", font_size=30, color=GREY),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(translator_eq, DOWN, buff=0.5)

        self.play(Write(translator_eq), run_time=1.5)
        self.play(FadeIn(where, shift=UP * 0.2), run_time=1)
        self.wait(1.5)

        # Key insight: identity initialization
        init_box = SurroundingRectangle(
            VGroup(
                MathTex(r"W_\ell^{(0)} = I_d", font_size=36),
                MathTex(r"b_\ell^{(0)} = \mathbf{0}", font_size=36),
            ).arrange(RIGHT, buff=1),
            color=GOLD, buff=0.3, corner_radius=0.1,
        )
        init_content = VGroup(
            MathTex(r"W_\ell^{(0)} = I_d", font_size=36, color=GOLD),
            MathTex(r"b_\ell^{(0)} = \mathbf{0}", font_size=36, color=GOLD),
        ).arrange(RIGHT, buff=1).move_to(init_box)

        init_label = Text(
            "Identity init = starts from logit lens!",
            font_size=22, color=GOLD,
        ).next_to(init_box, DOWN, buff=0.2)

        init_group = VGroup(init_box, init_content, init_label).next_to(where, DOWN, buff=0.6)
        # Rebuild box around content
        init_box.move_to(init_content)
        init_label.next_to(init_box, DOWN, buff=0.2)

        self.play(
            Create(init_box),
            Write(init_content[0]), Write(init_content[1]),
            run_time=1.5,
        )
        self.play(FadeIn(init_label), run_time=0.8)
        self.wait(2)

        # Full pipeline
        self.play(FadeOut(VGroup(idea, translator_eq, where, init_box, init_content, init_label)))

        full_eq = MathTex(
            r"p_\text{lens}^\ell",
            r"= \text{softmax}\Big(",
            r"W_U \cdot \text{LN}_\text{final}\big(",
            r"T_\ell(h_\ell)",
            r"\big) + b_U",
            r"\Big)",
            font_size=36,
        ).next_to(heading, DOWN, buff=1)
        full_eq[0].set_color(ACCENT)
        full_eq[3].set_color(GOLD)

        self.play(Write(full_eq), run_time=2)
        self.wait(1)

        # Loss function
        loss_eq = MathTex(
            r"\mathcal{L}_\ell = D_\text{KL}\!\left(",
            r"p_\text{lens}^\ell",
            r"\;\|\;\,",
            r"p_\text{final}",
            r"\right)",
            font_size=42, color=WHITE,
        ).next_to(full_eq, DOWN, buff=0.8)
        loss_eq[1].set_color(ACCENT)
        loss_eq[3].set_color(GREEN)

        loss_label = Text(
            "Minimize KL divergence from final-layer output",
            font_size=22, color=GREY,
        ).next_to(loss_eq, DOWN, buff=0.3)

        self.play(Write(loss_eq), run_time=1.5)
        self.play(FadeIn(loss_label), run_time=0.8)
        self.wait(2.5)
        self.play(FadeOut(VGroup(heading, full_eq, loss_eq, loss_label)))

    # ── 5. Training Animation ──────────────────────────────────────────
    def training_scene(self):
        heading = Text("Training: Per-Layer Optimization", font_size=44, color=GOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading), run_time=0.8)

        # Training config
        config = VGroup(
            Text("Llama-3.2-3B  |  28 layers  |  d=3072", font_size=22, color=GREY),
            Text("Adam optimizer  |  lr=0.001  |  cosine schedule", font_size=22, color=GREY),
            Text("50 epochs  |  batch_size=512  |  8 train sequences", font_size=22, color=GREY),
        ).arrange(DOWN, buff=0.15).next_to(heading, DOWN, buff=0.4)
        self.play(FadeIn(config, shift=UP * 0.2), run_time=1)

        # Animate a loss curve converging
        ax = Axes(
            x_range=[0, 50, 10],
            y_range=[0, 0.5, 0.1],
            x_length=8, y_length=3.5,
            axis_config={"color": GREY, "include_numbers": True, "font_size": 20},
            tips=False,
        ).shift(DOWN * 0.8)
        x_lab = ax.get_x_axis_label("Epoch", font_size=24, direction=DOWN)
        y_lab = ax.get_y_axis_label("KL Loss", font_size=24, direction=LEFT)

        self.play(Create(ax), FadeIn(x_lab), FadeIn(y_lab), run_time=1)

        # Simulate loss curves for 3 representative layers
        def loss_curve(layer_kl_final, noise_scale=0.02):
            """Simulate a training curve that converges to layer_kl_final."""
            start = 0.4 + np.random.rand() * 0.05
            x = np.linspace(0, 50, 100)
            y = layer_kl_final + (start - layer_kl_final) * np.exp(-x / 8)
            y += np.random.randn(len(y)) * noise_scale * np.exp(-x / 15)
            return np.clip(y, 0, 0.5)

        np.random.seed(42)
        curves_data = [
            ("Layer 0", RED, 0.012),
            ("Layer 14", ORANGE, 0.005),
            ("Layer 26", GREEN, 0.001),
        ]

        legend_items = VGroup()
        for name, col, final_kl in curves_data:
            y_vals = loss_curve(final_kl)
            x_vals = np.linspace(0, 50, 100)
            points = [ax.c2p(x, y) for x, y in zip(x_vals, y_vals)]
            curve = VMobject(color=col, stroke_width=2.5)
            curve.set_points_smoothly(points)

            legend_dot = Dot(color=col, radius=0.06)
            legend_text = Text(f"{name} (KL={final_kl})", font_size=16, color=col)
            li = VGroup(legend_dot, legend_text).arrange(RIGHT, buff=0.15)
            legend_items.add(li)

            self.play(Create(curve), run_time=1.5)

        legend_items.arrange(DOWN, buff=0.15, aligned_edge=LEFT).to_corner(DR, buff=0.5)
        self.play(FadeIn(legend_items), run_time=0.5)
        self.wait(2)
        self.play(FadeOut(VGroup(heading, config, ax, x_lab, y_lab, legend_items, *self.mobjects)))

    # ── 6. Code Walkthrough ─────────────────────────────────────────────
    def code_walkthrough_scene(self):
        heading = Text("Implementation", font_size=48, color=GOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading), run_time=0.8)

        # Translator class
        code_translator = Code(
            code='''\
class TunedLensTranslator(nn.Module):
    """Per-layer affine: h_l -> h_tilde"""
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=True)
        nn.init.eye_(self.linear.weight)    # W = I
        nn.init.zeros_(self.linear.bias)    # b = 0

    def forward(self, x):
        return self.linear(x)  # Wx + b''',
            language="python",
            font_size=18,
            background="rectangle",
            background_stroke_color=ACCENT,
            insert_line_no=False,
            style="monokai",
        ).scale(0.85).next_to(heading, DOWN, buff=0.4).shift(LEFT * 0.5)

        label1 = Text("The Translator", font_size=24, color=ACCENT).next_to(code_translator, LEFT, buff=0.3).rotate(PI / 2)

        self.play(FadeIn(code_translator), FadeIn(label1), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(VGroup(code_translator, label1)))

        # Training loop
        code_train = Code(
            code='''\
# Training loop (per layer)
h = translator(acts_batch)        # Affine transform
normed = model.ln_final(h)        # Final layer norm
logits = normed @ W_U + b_U       # Unembed to vocab
log_probs = log_softmax(logits)   # Log probabilities

# KL(p_lens || p_final)
loss = F.kl_div(log_probs, target_log_probs,
                reduction="batchmean", log_target=True)
loss.backward()
optimizer.step()''',
            language="python",
            font_size=18,
            background="rectangle",
            background_stroke_color=GREEN,
            insert_line_no=False,
            style="monokai",
        ).scale(0.85).next_to(heading, DOWN, buff=0.5)

        label2 = Text("Training Loop", font_size=24, color=GREEN).next_to(code_train, LEFT, buff=0.3).rotate(PI / 2)

        self.play(FadeIn(code_train), FadeIn(label2), run_time=1.5)
        self.wait(2.5)
        self.play(FadeOut(VGroup(heading, code_train, label2)))

    # ── 7. Results ──────────────────────────────────────────────────────
    def results_scene(self):
        heading = Text("Results: Llama-3.2-3B on Mess-3 HMM", font_size=40, color=GOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading), run_time=0.8)

        # ── Plot 1: KL(final || tuned) by layer ──
        ax1 = Axes(
            x_range=[0, 27, 5],
            y_range=[0, 0.014, 0.004],
            x_length=5.5, y_length=3,
            axis_config={"color": GREY, "include_numbers": True, "font_size": 16},
            tips=False,
        ).shift(LEFT * 3.2 + DOWN * 0.5)
        ax1_title = Text("KL(final || tuned lens)", font_size=18, color=ACCENT).next_to(ax1, UP, buff=0.2)
        ax1_xlab = Text("Layer", font_size=16, color=GREY).next_to(ax1, DOWN, buff=0.2)

        points1 = [ax1.c2p(l, kl) for l, kl in zip(LAYERS, KL_FINAL_TUNED)]
        dots1 = VGroup(*[Dot(p, color=ACCENT, radius=0.04) for p in points1])
        line1 = VMobject(color=ACCENT, stroke_width=2)
        line1.set_points_smoothly(points1)

        # ── Plot 2: Top-1 agreement ──
        ax2 = Axes(
            x_range=[0, 27, 5],
            y_range=[0.96, 1.0, 0.01],
            x_length=5.5, y_length=3,
            axis_config={"color": GREY, "include_numbers": True, "font_size": 16},
            tips=False,
        ).shift(RIGHT * 3.2 + DOWN * 0.5)
        ax2_title = Text("Top-1 Agreement (tuned)", font_size=18, color=GREEN).next_to(ax2, UP, buff=0.2)
        ax2_xlab = Text("Layer", font_size=16, color=GREY).next_to(ax2, DOWN, buff=0.2)

        points2 = [ax2.c2p(l, t) for l, t in zip(LAYERS, TOP1_TUNED)]
        dots2 = VGroup(*[Dot(p, color=GREEN, radius=0.04) for p in points2])
        line2 = VMobject(color=GREEN, stroke_width=2)
        line2.set_points_smoothly(points2)

        self.play(
            Create(ax1), Create(ax2),
            FadeIn(ax1_title), FadeIn(ax2_title),
            FadeIn(ax1_xlab), FadeIn(ax2_xlab),
            run_time=1,
        )
        self.play(
            Create(line1), Create(line2),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in dots1], lag_ratio=0.03),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in dots2], lag_ratio=0.03),
            run_time=2,
        )

        # Annotations
        ann1 = VGroup(
            Text("Layer 26: KL = 0.001", font_size=14, color=GOLD),
            Text("(best reconstruction)", font_size=12, color=GREY),
        ).arrange(DOWN, buff=0.05).next_to(ax1.c2p(26, 0.001), UP, buff=0.2)

        ann2 = VGroup(
            Text("Layer 26: 99.3%", font_size=14, color=GOLD),
        ).next_to(ax2.c2p(26, 0.993), UP + LEFT, buff=0.15)

        self.play(FadeIn(ann1), FadeIn(ann2), run_time=0.8)
        self.wait(2.5)
        self.play(FadeOut(VGroup(
            heading, ax1, ax2, ax1_title, ax2_title, ax1_xlab, ax2_xlab,
            line1, line2, dots1, dots2, ann1, ann2,
        )))

        # ── Plot 3: Tuned vs Logit lens (KL from HMM) ──
        heading2 = Text("Tuned Lens vs Logit Lens", font_size=40, color=GOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading2), run_time=0.8)

        ax3 = Axes(
            x_range=[0, 27, 5],
            y_range=[0, 1.6, 0.4],
            x_length=9, y_length=4,
            axis_config={"color": GREY, "include_numbers": True, "font_size": 18},
            tips=False,
        ).shift(DOWN * 0.3)
        ax3_xlab = Text("Layer", font_size=20, color=GREY).next_to(ax3, DOWN, buff=0.3)
        ax3_ylab = Text("KL(HMM || lens)", font_size=20, color=GREY).next_to(ax3, LEFT, buff=0.3).rotate(PI / 2)

        # Tuned lens line (smooth, low)
        pts_tuned = [ax3.c2p(l, kl) for l, kl in zip(LAYERS, KL_HMM_TUNED)]
        line_tuned = VMobject(color=ACCENT, stroke_width=3)
        line_tuned.set_points_smoothly(pts_tuned)

        # Logit lens line (noisy, high) - skip None values
        valid_logit = [(l, kl) for l, kl in zip(LAYERS, KL_HMM_LOGIT) if kl is not None]
        pts_logit = [ax3.c2p(l, min(kl, 1.55)) for l, kl in valid_logit]
        line_logit = VMobject(color=RED, stroke_width=3)
        line_logit.set_points_smoothly(pts_logit)

        # Legend
        legend = VGroup(
            VGroup(
                Line(ORIGIN, RIGHT * 0.5, color=ACCENT, stroke_width=3),
                Text("Tuned Lens", font_size=18, color=ACCENT),
            ).arrange(RIGHT, buff=0.15),
            VGroup(
                Line(ORIGIN, RIGHT * 0.5, color=RED, stroke_width=3),
                Text("Logit Lens", font_size=18, color=RED),
            ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).to_corner(UR, buff=0.8)

        self.play(Create(ax3), FadeIn(ax3_xlab), FadeIn(ax3_ylab), run_time=1)
        self.play(Create(line_logit), FadeIn(legend[1]), run_time=1.5)
        self.play(Create(line_tuned), FadeIn(legend[0]), run_time=1.5)

        # Highlight the gap
        brace_x = 8
        tuned_y = KL_HMM_TUNED[brace_x]
        logit_y = KL_HMM_LOGIT[brace_x]
        brace = BraceBetweenPoints(
            ax3.c2p(brace_x, tuned_y), ax3.c2p(brace_x, logit_y),
            direction=RIGHT, color=GOLD,
        )
        brace_label = Text("Tuned lens\nrecovers info!", font_size=16, color=GOLD).next_to(brace, RIGHT, buff=0.15)

        self.play(Create(brace), FadeIn(brace_label), run_time=1)
        self.wait(2.5)
        self.play(FadeOut(VGroup(
            heading2, ax3, ax3_xlab, ax3_ylab,
            line_tuned, line_logit, legend, brace, brace_label,
        )))

    # ── 8. Key Insights ─────────────────────────────────────────────────
    def key_insights_scene(self):
        heading = Text("Key Insights", font_size=48, color=GOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading), run_time=0.8)

        insights = [
            ("1.", "Even layer 0 achieves 97% top-1 agreement", "with just an affine correction.", ACCENT),
            ("2.", "KL(final || tuned) drops from 0.012 to 0.001", "across layers — later layers do genuine new computation.", GREEN),
            ("3.", "Belief-state R² peaks at layer 2 (0.999)", "but tuned lens keeps improving — representations re-encode.", PURPLE),
            ("4.", "Logit lens fails at early layers (KL > 1.5)", "but tuned lens stays below 0.05 everywhere.", RED),
        ]

        items = VGroup()
        for num, line1, line2, col in insights:
            item = VGroup(
                Text(num, font_size=28, color=col, weight=BOLD),
                VGroup(
                    Text(line1, font_size=22, color=WHITE),
                    Text(line2, font_size=20, color=GREY),
                ).arrange(DOWN, buff=0.05, aligned_edge=LEFT),
            ).arrange(RIGHT, buff=0.3, aligned_edge=UP)
            items.add(item)

        items.arrange(DOWN, buff=0.4, aligned_edge=LEFT).next_to(heading, DOWN, buff=0.6).shift(LEFT * 1)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=1)
            self.wait(0.8)

        self.wait(2)

        # Closing
        self.play(FadeOut(VGroup(heading, items)))
        thanks = Text("Tuned Lens", font_size=64, color=ACCENT)
        sub = MathTex(
            r"T_\ell(h_\ell) \;\xrightarrow{\text{LN} + W_U}\; p_\text{lens}^\ell \approx p_\text{final}",
            font_size=36, color=WHITE,
        ).next_to(thanks, DOWN, buff=0.5)
        self.play(Write(thanks), run_time=1)
        self.play(Write(sub), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(VGroup(thanks, sub)))
