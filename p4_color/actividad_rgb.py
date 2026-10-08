import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Directory Check
# ================================================================================================
Path("p4_color/resultados").mkdir(parents=True, exist_ok=True)

# Load Image Bars
# ================================================================================================
img_bars = Image.open("p4_color/datos/barras_referencia.png").convert("RGB")
img_bars_captured = Image.open("p4_color/datos/barras_capturadas.png").convert("RGB")

# Convert Image - Array
# ================================================================================================
arr_bars = np.array(img_bars, dtype=np.float64)
arr_bars_captured = np.array(img_bars_captured, dtype=np.float64)

# Check Image
# ================================================================================================
print(f"Array Shape (Reference): {arr_bars.shape}")
print(f"Array Shape (Captured): {arr_bars_captured.shape}")

# Separate into RGB components
# ================================================================================================
# ===== Reference ===== #
arr_bars_R = arr_bars[:,:,0]
arr_bars_G = arr_bars[:,:,1]
arr_bars_B = arr_bars[:,:,2]

# ===== Capture ===== #
arr_bars_captured_R = arr_bars_captured[:,:,0]
arr_bars_captured_G = arr_bars_captured[:,:,1]
arr_bars_captured_B = arr_bars_captured[:,:,2]

# Check Sizes
# ================================================================================================
# ===== Reference ===== #
print(f"Array-R Shape (Ref): {arr_bars_R.shape}")
print(f"Array-G Shape (Ref): {arr_bars_G.shape}")
print(f"Array-B Shape (Ref): {arr_bars_B.shape}")

# ===== Capture ===== #
print(f"Array-R Shape (Cap): {arr_bars_captured_R.shape}")
print(f"Array-G Shape (Cap): {arr_bars_captured_G.shape}")
print(f"Array-B Shape (Cap): {arr_bars_captured_B.shape}")

# Visualize 3 Channels (Reference)
# ================================================================================================
fig, axes = plt.subplots(1, 3, figsize = (12, 4))
axes[0].imshow(arr_bars_R, cmap = "gray", vmin = 0, vmax = 255)
axes[0].set_title("R Channel")

axes[1].imshow(arr_bars_G, cmap = "gray", vmin = 0, vmax = 255)
axes[1].set_title("G Channel")

axes[2].imshow(arr_bars_B, cmap = "gray", vmin = 0, vmax = 255)
axes[2].set_title("B Channel")

plt.tight_layout()
plt.savefig('p4_color/resultados/channels_bars_ref.png', dpi=100, bbox_inches='tight')

# Visualize 3 Channels (Capture)
# ================================================================================================
fig1, axes1 = plt.subplots(1, 3, figsize = (12, 4))
axes1[0].imshow(arr_bars_captured_R, cmap = "gray", vmin = 0, vmax = 255)
axes1[0].set_title("R Channel")

axes1[1].imshow(arr_bars_captured_G, cmap = "gray", vmin = 0, vmax = 255)
axes1[1].set_title("G Channel")

axes1[2].imshow(arr_bars_captured_B, cmap = "gray", vmin = 0, vmax = 255)
axes1[2].set_title("B Channel")

plt.tight_layout()
plt.savefig('p4_color/resultados/channels_bars_cap.png', dpi=100, bbox_inches='tight')

# Reorder Images
# ================================================================================================
# ===== Reference ===== #
new_bars_ref = np.stack([arr_bars_B, arr_bars_G, arr_bars_R], axis = 2)
new_bars_ref = new_bars_ref.astype(np.uint8)

# ===== Capture ===== #
new_bars_cap = np.stack([arr_bars_captured_B, arr_bars_captured_G, arr_bars_captured_R], axis = 2)
new_bars_cap = new_bars_cap.astype(np.uint8)

# Show Changes When Reconstructing
# ================================================================================================
# ===== Reference ===== #
fig2, axes2 = plt.subplots(1, 2, figsize = (12, 4))
axes2[0].imshow(img_bars)
axes2[0].set_title("Original Reference Image")

axes2[1].imshow(new_bars_ref)
axes2[1].set_title("Reconstructed Reference Image (BGR)")

plt.tight_layout()
plt.savefig('p4_color/resultados/reconstructed_bars_ref.png', dpi=100, bbox_inches='tight')

# ===== Capture ===== #
fig3, axes3 = plt.subplots(1, 2, figsize = (12, 4))
axes3[0].imshow(img_bars_captured)
axes3[0].set_title("Original Captured Image")

axes3[1].imshow(new_bars_cap)
axes3[1].set_title("Reconstructed Captured Image (BGR)")

plt.tight_layout()
plt.savefig('p4_color/resultados/reconstructed_bars_cap.png', dpi=100, bbox_inches='tight')

# ===== Comparison ===== #
fig4, axes4 = plt.subplots(1, 2, figsize = (12, 4))
axes4[0].imshow(new_bars_ref)
axes4[0].set_title("Reconstructed Reference Image (BGR)")

axes4[1].imshow(new_bars_cap)
axes4[1].set_title("Reconstructed Captured Image (BGR)")
plt.tight_layout()
plt.savefig('p4_color/resultados/reconstructed_bars_comparison.png', dpi=100, bbox_inches='tight')

plt.show()