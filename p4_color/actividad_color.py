from p4_color.conversion import rgb2ycrcb, yrcb2rgb
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
plt.savefig('p4_color/resultados/yq_crq_cbq_bars_ref.png', dpi=100, bbox_inches='tight')

'''
V1 = int(Yq[141, 91])
V2 = int(Yq[265, 1856])
V3 = int(Crq[141, 91])
V4 = int(Crq[443, 1323])
'''
#print(f"V1 = {V1}, V2 = {V2}, V3 = {V3}, V4 = {V4}")
img_bars_rec = yrcb2rgb(Yq, Crq, Cbq)
diff = arr_bars.astype(np.float64) - img_bars_rec.astype(np.float64)  
 

RMSE = np.mean(diff**2)

diff = np.clip(np.abs(diff*20 + 128), 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(arr_bars); axes[0].set_title('RGB original')
axes[1].imshow(img_bars_rec); axes[1].set_title('RGB reconstructido')
axes[2].imshow(diff); axes[2].set_title('Diferencia x20 + 128')

plt.tight_layout()
plt.savefig('p4_color/resultados/rgb_rec_bars_ref.png', dpi=100, bbox_inches='tight')



print(f"RMSE = {RMSE:.4f}")

plt.show()