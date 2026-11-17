import os
from PIL import Image, ImageDraw, ImageFont

SRC_IMG = "/Users/vdevanarayanan/.gemini/antigravity/brain/8b221b9a-6b22-482c-9bb4-5eefcef13e37/.user_uploaded/media_1794210268130_3a1ac80a.jpg"
OUT_IMG = "plots/model_architecture_edited.png"
os.makedirs("plots", exist_ok=True)

# Load base image
base = Image.open(SRC_IMG).convert("RGBA")
W, H = base.size

# Overlay for alpha transparency
overlay = Image.new("RGBA", (W, H), (255, 255, 255, 0))
draw = ImageDraw.Draw(overlay)

# Fonts
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

font_title = ImageFont.truetype(FONT_BOLD_PATH, 19)
font_sub = ImageFont.truetype(FONT_REG_PATH, 13)
font_badge = ImageFont.truetype(FONT_BOLD_PATH, 11)
font_badge_sm = ImageFont.truetype(FONT_BOLD_PATH, 10)
font_output_h = ImageFont.truetype(FONT_BOLD_PATH, 18)
font_output_sub = ImageFont.truetype(FONT_BOLD_PATH, 13)

# 1. Dim & Strike-out Multi-label head (X: 369 to 651, Y: 655 to 740)
draw.rounded_rectangle([368, 655, 652, 740], radius=8, fill=(254, 226, 226, 175), outline=(220, 38, 38, 255), width=2)
# Cross lines
draw.line([370, 657, 650, 738], fill=(220, 38, 38, 200), width=3)
draw.line([370, 738, 650, 657], fill=(220, 38, 38, 200), width=3)
# Badge
draw.rounded_rectangle([400, 680, 620, 715], radius=5, fill=(220, 38, 38, 240), outline=(185, 28, 28, 255), width=1)
draw.text((510, 692), "✕ EXCISED / REMOVED", font=font_badge, fill="white", anchor="mm")
draw.text((510, 705), "Eliminated gradient dilution", font=font_badge_sm, fill="#fee2e2", anchor="mm")

# 2. Dim & Strike-out Rationale head (X: 680 to 961, Y: 655 to 740)
draw.rounded_rectangle([679, 655, 962, 740], radius=8, fill=(254, 226, 226, 175), outline=(220, 38, 38, 255), width=2)
# Cross lines
draw.line([681, 657, 960, 738], fill=(220, 38, 38, 200), width=3)
draw.line([681, 738, 960, 657], fill=(220, 38, 38, 200), width=3)
# Badge
draw.rounded_rectangle([710, 680, 930, 715], radius=5, fill=(220, 38, 38, 240), outline=(185, 28, 28, 255), width=1)
draw.text((820, 692), "✕ EXCISED / REMOVED", font=font_badge, fill="white", anchor="mm")
draw.text((820, 705), "Eliminated sparse multi-task penalty", font=font_badge_sm, fill="#fee2e2", anchor="mm")

# Dim arrows coming from excised heads to Final Output (Y: 740 to 805)
draw.rectangle([480, 742, 540, 804], fill=(255, 255, 255, 210))
draw.line([510, 742, 510, 804], fill=(220, 38, 38, 120), width=2)
draw.rectangle([790, 742, 850, 804], fill=(255, 255, 255, 210))
draw.line([820, 742, 820, 804], fill=(220, 38, 38, 120), width=2)

# 3. Enhance Binary Head (X: 59 to 340, Y: 655 to 740)
# Glowing highlight
draw.rounded_rectangle([57, 653, 342, 742], radius=10, outline=(16, 185, 129, 255), width=3)
draw.rounded_rectangle([65, 638, 335, 658], radius=5, fill=(16, 185, 129, 250), outline=(5, 150, 105, 255), width=1)
draw.text((200, 648), "★ ACTIVE PURE BINARY HEAD", font=font_badge, fill="white", anchor="mm")

# Emphasize active arrow from Binary Head to Final Output
draw.line([200, 740, 200, 805], fill=(16, 185, 129, 255), width=3)
# Arrow head pointing down to Final Output
draw.polygon([(194, 797), (206, 797), (200, 807)], fill=(16, 185, 129, 255))

# 4. RoBERTa text enhancement badge (X: 292 to 497, Y: 220 to 290)
draw.rounded_rectangle([295, 206, 494, 224], radius=4, fill=(43, 122, 102, 240))
draw.text((395, 215), "✓ Top 6 Layers Fine-Tuned (lr=3e-5)", font=font_badge_sm, fill="white", anchor="mm")

# 5. ViT visual stream badge (X: 526 to 731, Y: 220 to 290)
draw.rounded_rectangle([530, 206, 727, 224], radius=4, fill=(179, 93, 73, 240))
draw.text((628, 215), "❄ Frozen & Pre-Cached Patches", font=font_badge_sm, fill="white", anchor="mm")

# 6. SDPA Acceleration badge on Cross-Attention Box (X: 59 to 964, Y: 370 to 450)
draw.rounded_rectangle([680, 355, 960, 377], radius=5, fill=(91, 75, 145, 255), outline=(139, 92, 246, 255), width=1)
draw.text((820, 366), "⚡ SDPA PyTorch 2.0 (330x Speedup)", font=font_badge, fill="white", anchor="mm")

# 7. Context Gate badge (X: 59 to 964, Y: 515 to 595)
draw.rounded_rectangle([610, 500, 960, 522], radius=5, fill=(91, 75, 145, 255), outline=(139, 92, 246, 255), width=1)
draw.text((785, 511), "★ Class-Weighted Dynamic Gating (T=0.71)", font=font_badge, fill="white", anchor="mm")

# 8. MultiOFF Direct Text Bypass Pathway (drawn as an orange dashed/solid routing channel)
# Starts from RoBERTa text box left side (X: 290, Y: 255)
# Paths out to X: 35, curves down to Y: 698, and enters Binary Head left side (X: 57, Y: 698)
bypass_color = (234, 88, 12, 255) # Bright orange
draw.line([(290, 255), (35, 255), (35, 698), (57, 698)], fill=bypass_color, width=3)
# Arrow head entering Binary Head
draw.polygon([(49, 692), (49, 704), (58, 698)], fill=bypass_color)

# Label on MultiOFF Bypass channel
draw.rounded_rectangle([8, 435, 48, 560], radius=5, fill=(234, 88, 12, 245))
# Write text vertically or badge
draw.rounded_rectangle([15, 340, 250, 365], radius=5, fill=(234, 88, 12, 250), outline=(194, 65, 12, 255), width=1)
draw.text((132, 352), "MultiOFF Route: Unimodal Bypass", font=font_badge, fill="white", anchor="mm")

draw.rounded_rectangle([15, 605, 230, 627], radius=5, fill=(234, 88, 12, 250), outline=(194, 65, 12, 255), width=1)
draw.text((122, 616), "MultiOFF Result: 95.30% Acc | 0.9824 AUC", font=font_badge_sm, fill="white", anchor="mm")

# 9. Update Final Output Box (X: 59 to 964, Y: 805 to 885)
# Clear original text inside the box with matching background fill #ede9e3
draw.rounded_rectangle([62, 808, 961, 882], radius=10, fill=(244, 240, 235, 255), outline=(16, 185, 129, 255), width=2)
draw.text((512, 827), "Final Pure Binary Decision Output (Clean vs. Harmful / Offensive)", font=font_output_h, fill="#1e293b", anchor="mm")
draw.text((512, 848), "► MultiOFF: 95.30% Accuracy | 95.07% Macro F1 | 0.9824 AUC  (Exceeded >70% target by +25.3%)", font=font_output_sub, fill="#c2410c", anchor="mm")
draw.text((512, 866), "► HarMeme: 83.62% Accuracy | 81.57% Macro F1 | 0.8710 AUC  (Matches Published Academic SOTA)", font=font_output_sub, fill="#047857", anchor="mm")

# 10. Updated Legend at bottom
draw.rectangle([0, 895, W, H], fill=(255, 255, 255, 255))
legend_text = "Orange: MultiOFF Direct Text Route (95.30%)   |   Purple: HarMeme SDPA Multimodal Route (83.62%)   |   Green: Active Pure Binary Head   |   Red: Excised Multi-Task Heads"
draw.text((W // 2, 915), legend_text, font=font_sub, fill="#475569", anchor="mm")

# Merge overlay onto base
final_img = Image.alpha_composite(base, overlay)
final_img.convert("RGB").save(OUT_IMG, dpi=(300, 300))
print(f"[✓] Saved edited model diagram to: {OUT_IMG}")

