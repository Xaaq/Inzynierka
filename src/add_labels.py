import glob
import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# plt.figure()
# plt.xlabel("Iteracja ucząca")
# # plt.ylabel("[s²] Błąd średniokwadratowy [s²]")
# plt.ylabel("Błąd średniokwadratowy")
#
# plt.plot([1, 2, 3, 1500], [0, 7, 8, 9], label="Dane treningowe")
# plt.plot([1, 2, 3, 1500], [6, 10, 11, 20], label="Dane testowe")
# plt.ylim([0, 400])
# plt.legend()
#
# # plt.savefig("/home/xaaq/my-projects/inzynierka/src/figura.png", bbox_inches="tight")
# plt.show()

image = Image.open("/home/xaaq/my-projects/inzynierka/src/figura.png")
image =np.array(image)

a = image[105:133, 10:30, :]

for i in glob.glob("/home/xaaq/my-projects/inzynierka/inzynierka_latex/wykresy/*/*/*.png"):
    image2 = Image.open(i)
    image2 = np.array(image2)
    image2[105:133, 10:30, :] = a
    Image.fromarray(image2).save(i)