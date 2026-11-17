import matplotlib
import numpy as np


def feld_zu_bild(feld, wand, farben, vmin, vmax, vergroesserung=4):
  # 1. Werte auf ein Maß von 0 bis 1 bringen
  normiert = (feld - vmin) / (vmax - vmin)
  normiert = np.clip(normiert, 0, 1)

  # 2. Farben aus der Farbtabelle zuweisen (Rot, Blau etc.)
  farbtabelle = matplotlib.colormaps[farben]
  bild = farbtabelle(normiert)[:, :, :3]

  # 3. Wände und den Flügel in einem schönen Grau einfärben
  bild[wand == 0] = [0.5, 0.5, 0.5]

  # 4. Bild drehen und spiegeln, damit es richtig herum liegt
  bild = np.transpose(bild, (1, 0, 2))
  bild = bild[::-1, :, :]

  # 5. Pixel vergrößern, damit es scharf aussieht
  bild = np.repeat(bild, vergroesserung, axis=0)
  bild = np.repeat(bild, vergroesserung, axis=1)

  # 6. In echte Bilddaten (0 bis 255) umwandeln
  return (bild * 255).astype(np.uint8)

