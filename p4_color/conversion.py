import numpy as np

def rgb2ycrcb(I_RGB):
    # Input: uint8 NumPy array with shape (height, width, 3)
    # Output: three uint8 arrays — Yq, Crq, Cbq

    I = I_RGB.astype(np.float64)

    # Channel Separation and Normalization
    # ============================================================================================
    R = I[:,:,0]/255.0
    G = I[:,:,1]/255.0
    B = I[:,:,2]/255.0

    # Luma Calculation
    # ============================================================================================
    Y = 0.299*R + 0.587*G + 0.114*B

    # Difference Signals
    # ============================================================================================
    R_Y = R - Y
    B_Y = B - Y

    # Chrominance Signals
    # ============================================================================================
    Cr = 0.713*R_Y
    Cb = 0.564*B_Y

    # Quantized Signals
    # ============================================================================================
    Yq = np.clip(16 + np.round(219*Y), 0, 255).astype(np.uint8)
    Crq = np.clip(128 + np.round(224*Cr), 0, 255).astype(np.uint8)
    Cbq = np.clip(128 + np.round(224*Cb), 0, 255).astype(np.uint8)

    return Yq, Crq, Cbq
