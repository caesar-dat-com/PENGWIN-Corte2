"""CLAHE experimental sobre la ventana normalizada; no modifica el CT en HU."""
import cv2,numpy as np,torch

def clahe_batch(x):
    transform=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8))
    images=np.stack([transform.apply(np.rint(im[0].numpy()*255).clip(0,255).astype(np.uint8)).astype(np.float32)/255 for im in x])
    return torch.from_numpy(images)[:,None].repeat(1,3,1,1)
