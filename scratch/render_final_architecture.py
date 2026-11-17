import os
import matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_IMG = "plots/model_architecture.png"
os.makedirs("plots", exist_ok=True)

# -----------------------------------------------------------------
# High-DPI Figure Setup (matching clean publication style)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 15.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# -----------------------------------------------------------------
# Color Palette (matching user reference diagram aesthetic)
# -----------------------------------------------------------------
COLOR_BG_INPUT = "#f4f0eb"
COLOR_BORDER_INPUT = "#7d756d"
COLOR_TEXT_INPUT = "#2b2723"

COLOR_TEAL_BG = "#eaf6f3"
COLOR_TEAL_BORDER = "#2b7a66"
COLOR_TEAL_TEXT = "#164e42"

COLOR_CORAL_BG = "#fbf1ee"
COLOR_CORAL_BORDER = "#b35d49"
COLOR_CORAL_TEXT = "#6c2818"

COLOR_PURPLE_BG = "#f2eff9"
COLOR_PURPLE_BORDER = "#635396"
COLOR_PURPLE_TEXT = "#382967"

COLOR_HEAD_BG = "#f0f9f5"
COLOR_HEAD_BORDER = "#1b6d58"
COLOR_HEAD_TEXT = "#0b3b2e"

# Helper for rounded cards
def draw_card(ax, xy, w, h, title, subtitle="", bg=COLOR_TEAL_BG, border=COLOR_TEAL_BORDER, text_col=COLOR_TEAL_TEXT, radius=2.2, lw=1.6, title_size=11.5, sub_size=8.8):
    x, y = xy
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.35,rounding_size={radius}",
                                facecolor=bg, edgecolor=border, linewidth=lw, zorder=2)
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=text_col, zorder=3)
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=sub_size, color=text_col, alpha=0.92, zorder=3)
    else:
        ax.text(x + w/2, y + h*0.5, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=text_col, zorder=3)

# Helper for directional flow arrows with optional label
def draw_arrow(ax, p1, p2, color="#64748b", lw=1.8, dashed=False, label="", label_pos=0.5, label_offset=(0, 0)):
    ls = "--" if dashed else "-"
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="->,head_width=0.34,head_length=0.48",
                                color=color, lw=lw, linestyle=ls,
                                shrinkA=2, shrinkB=2), zorder=1)
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos + label_offset[0]
        my = p1[1] + (p2[1] - p1[1]) * label_pos + label_offset[1]
        ax.text(mx, my, label, fontsize=8.0, fontweight="bold", color=color, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=0.8, alpha=0.95))

# Helper for step badges on the left margin
def draw_step_badge(ax, y_center, step_num, step_name):
    ax.text(5.5, y_center + 1.2, f"STEP {step_num}", fontsize=7.5, fontweight="bold", color="#786f68", ha="center", va="center")
    ax.text(5.5, y_center - 1.2, step_name, fontsize=7.2, color="#9c938c", ha="center", va="center")

# -----------------------------------------------------------------
# 1. Top Card: Meme Input
# -----------------------------------------------------------------
draw_card(ax, (10, 89.5), 80, 7.5, "Meme input", "image  +  caption  +  surrounding post text",
          bg=COLOR_BG_INPUT, border=COLOR_BORDER_INPUT, text_col=COLOR_TEXT_INPUT, title_size=13, sub_size=9.5)

# -----------------------------------------------------------------
# 2. STEP 1: OCR Extraction (Prerequisite for in-image text)
# -----------------------------------------------------------------
draw_step_badge(ax, 79.5, "1", "OCR Extraction")

# OCR extract card sits on the left between Meme Input and RoBERTa
draw_card(ax, (10, 76.0), 16, 7.2, "OCR extract", "extracts in-image text",
          bg=COLOR_TEAL_BG, border=COLOR_TEAL_BORDER, text_col=COLOR_TEAL_TEXT, title_size=10.8, sub_size=8.0)

# Meme Input (Image) -> OCR extract
draw_arrow(ax, (18.0, 89.5), (18.0, 83.2), color=COLOR_TEAL_BORDER, lw=1.8, label="meme image", label_pos=0.5)

# -----------------------------------------------------------------
# 3. STEP 2: The Three Deep Encoders (RoBERTa, ViT, Post RoBERTa)
# -----------------------------------------------------------------
draw_step_badge(ax, 65.5, "2", "Deep Encoders")

# 3A. RoBERTa Text (Left: width 26, x=10 to 36)
draw_card(ax, (10, 62.0), 26, 7.5, "RoBERTa text", "caption  +  OCR in-image text",
          bg=COLOR_TEAL_BG, border=COLOR_TEAL_BORDER, text_col=COLOR_TEAL_TEXT, title_size=11.5, sub_size=8.5)

# OCR extracted text -> RoBERTa text
draw_arrow(ax, (18.0, 76.0), (18.0, 69.5), color=COLOR_TEAL_BORDER, lw=2.0, label="OCR text", label_pos=0.5)

# Caption directly from Meme Input -> RoBERTa text (bypasses OCR, flows cleanly on right of OCR)
draw_arrow(ax, (31.0, 89.5), (31.0, 69.5), color=COLOR_TEAL_BORDER, lw=1.8, label="caption", label_pos=0.42)

# 3B. ViT Visual (Center: width 26, x=40 to 66)
draw_card(ax, (40, 62.0), 26, 7.5, "ViT visual", "image patches (224×224 RGB)",
          bg=COLOR_CORAL_BG, border=COLOR_CORAL_BORDER, text_col=COLOR_CORAL_TEXT, title_size=11.5, sub_size=8.5)

# Raw image from Meme Input -> ViT visual
draw_arrow(ax, (53.0, 89.5), (53.0, 69.5), color=COLOR_CORAL_BORDER, lw=1.8, label="raw image pixels", label_pos=0.42)

# 3C. Post RoBERTa (Right: width 20, x=70 to 90)
draw_card(ax, (70, 62.0), 20, 7.5, "Post RoBERTa", "post text",
          bg=COLOR_PURPLE_BG, border=COLOR_PURPLE_BORDER, text_col=COLOR_PURPLE_TEXT, title_size=11.5, sub_size=8.5)

# Surrounding post text from Meme Input -> Post RoBERTa
draw_arrow(ax, (80.0, 89.5), (80.0, 69.5), color=COLOR_PURPLE_BORDER, lw=1.8, label="post text", label_pos=0.42)

# -----------------------------------------------------------------
# 4. STEP 3: Bidirectional Cross-Attention (DDCAF Core)
# -----------------------------------------------------------------
draw_step_badge(ax, 52.0, "3", "Cross-Attention")

# Cross-attention fuses text sequence tokens and visual patch tokens (x=10 to 66)
draw_card(ax, (10, 48.0), 56, 8.5, "Bidirectional cross-attention", "TGVA + VGTA fusion core (DDCAF with SDPA)",
          bg=COLOR_PURPLE_BG, border=COLOR_PURPLE_BORDER, text_col=COLOR_PURPLE_TEXT, title_size=12.0, sub_size=9.0)

# RoBERTa text representations -> Cross-Attention
draw_arrow(ax, (23.0, 62.0), (28.0, 56.5), color=COLOR_TEAL_BORDER, lw=2.0)

# ViT visual patch representations -> Cross-Attention
draw_arrow(ax, (53.0, 62.0), (48.0, 56.5), color=COLOR_CORAL_BORDER, lw=2.0)

# -----------------------------------------------------------------
# 5. STEP 4: Dynamic Gating Fusion (Context Gate)
# -----------------------------------------------------------------
draw_step_badge(ax, 38.0, "4", "Context Gate")

draw_card(ax, (10, 34.0), 80, 8.5, "Dynamic gating fusion", "context gate balances modalities & suppresses noisy post text",
          bg=COLOR_PURPLE_BG, border=COLOR_PURPLE_BORDER, text_col=COLOR_PURPLE_TEXT, title_size=12.2, sub_size=9.2)

# Cross-attention fused features -> Dynamic Gating
draw_arrow(ax, (38.0, 48.0), (38.0, 42.5), color=COLOR_PURPLE_BORDER, lw=2.4)

# Post RoBERTa stream -> Dynamic Gating directly (bypasses cross-attention to act as context prior)
draw_arrow(ax, (80.0, 62.0), (80.0, 42.5), color=COLOR_PURPLE_BORDER, lw=1.8, dashed=True, label="post context", label_pos=0.5)

# -----------------------------------------------------------------
# 6. STEP 5: Pure Centered Binary Head (Single Head Architecture)
# -----------------------------------------------------------------
draw_step_badge(ax, 24.0, "5", "Binary Head")

# Centered prominently below Dynamic Gating (no crossed-out heads!)
draw_card(ax, (28, 20.0), 44, 8.5, "Binary head", "harmful / offensive  vs.  clean",
          bg=COLOR_HEAD_BG, border=COLOR_HEAD_BORDER, text_col=COLOR_HEAD_TEXT, title_size=12.8, sub_size=9.5, lw=2.2)

# Dynamic Gating -> Binary Head
draw_arrow(ax, (50.0, 34.0), (50.0, 28.5), color=COLOR_HEAD_BORDER, lw=2.4)

# -----------------------------------------------------------------
# 7. STEP 6: Final Output
# -----------------------------------------------------------------
draw_step_badge(ax, 10.2, "6", "Prediction")

draw_card(ax, (10, 6.5), 80, 7.8, "Final output", "binary label: clean  vs.  harmful / offensive",
          bg=COLOR_BG_INPUT, border=COLOR_BORDER_INPUT, text_col=COLOR_TEXT_INPUT, title_size=12.5, sub_size=9.5)

# Binary Head -> Final Output
draw_arrow(ax, (50.0, 20.0), (50.0, 14.3), color=COLOR_BORDER_INPUT, lw=2.4)

# -----------------------------------------------------------------
# 8. Legend at bottom
# -----------------------------------------------------------------
legend_text = "Teal = text stream & OCR       Coral = visual stream       Purple = fusion & context gate       Green = pure binary head"
ax.text(50, 2.2, legend_text, ha="center", va="center", fontsize=9.0, color="#59524c", style="italic")

plt.tight_layout()
plt.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[✓] Rendered clean, uncrossed model architecture diagram with OCR precedence to: {OUT_IMG}")
