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
    #Y = 0.2126*R + 0.7152*G + 0.0722*B
    # Difference Signals
    # ============================================================================================
    R_Y = R - Y
    B_Y = B - Y

    # Chrominance Signals
    # ============================================================================================
    Cr = 0.713*R_Y
    #Cr = 0.635*R_Y
    Cb = 0.564*B_Y
    #Cb = 0.539*B_Y

    # Quantized Signals
    # ============================================================================================
    Yq = np.clip(16 + np.round(219*Y), 0, 255).astype(np.uint8)
    Crq = np.clip(128 + np.round(224*Cr), 0, 255).astype(np.uint8)
    Cbq = np.clip(128 + np.round(224*Cb), 0, 255).astype(np.uint8)

    return Yq, Crq, Cbq

def yrcb2rgb(Yq, Crq, Cbq):
    # Input: three uint8 arrays — Yq, Crq, Cbq
    # Output: uint8 NumPy array with shape (height, width, 3)

    # Normalization
    # ============================================================================================
    Y = (Yq.astype(np.float64) - 16)/219.0
    Cr = (Crq.astype(np.float64) - 128)/224.0
    Cb = (Cbq.astype(np.float64) - 128)/224.0

    R_Y = Cr/0.713
    B_Y = Cb/0.564

    # RGB Calculation
    # ============================================================================================
    R = Y + R_Y
    B = Y + B_Y
    G = (Y - 0.299*R - 0.114*B)/0.587
    

    # Clipping and Conversion to uint8
    # ============================================================================================
    R_u8 = np.clip(np.round(R * 255), 0, 255).astype(np.uint8)
    G_u8 = np.clip(np.round(G * 255), 0, 255).astype(np.uint8)
    B_u8 = np.clip(np.round(B * 255), 0, 255).astype(np.uint8)

    I_RGB = np.stack([R_u8, G_u8, B_u8], axis=2)

    return I_RGB