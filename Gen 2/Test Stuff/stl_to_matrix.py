import pptk
import numpy as np
import pandas as pd

point_cloud = np.loadtxt("C:/Users/mihir/Desktop/Mihir/Science Fair/Multiple Sclerosis AI/Gen 2/Test/optimum_pride.asc")
point_cloud.shape
print(point_cloud)

v = pptk.viewer(point_cloud)