from p4_color.conversion import rgb2ycrcb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_bars = Image.open("p4_color/datos/barras_referencia.png").convert("RGB")
arr_bars = np.array(img_bars)

Yq, Crq, Cbq = rgb2ycrcb(arr_bars)
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(arr_bars); axes[0].set_title('RGB original')
axes[1].imshow(Yq,  cmap='gray', vmin=0, vmax=255); axes[1].set_title('Yq')
axes[2].imshow(Crq, cmap='gray', vmin=0, vmax=255); axes[2].set_title('Crq')
axes[3].imshow(Cbq, cmap='gray', vmin=0, vmax=255); axes[3].set_title('Cbq')
plt.tight_layout()
plt.savefig('p4_color/resultados/yq_crq_cbq_bars.png', dpi=100, bbox_inches='tight')
plt.show()