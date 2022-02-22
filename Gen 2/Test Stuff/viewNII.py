import nibabel
import matplotlib.pyplot as plt
import pandas as pd

#struct = nibabel.load("C:/Users/mihir/Desktop/Mihir/Science Fair/Multiple Sclerosis AI/Gen 2/Test Stuff/brainweb.mnc")
#struct_arr = struct.get_fdata()

#nibabel.viewers.OrthoSlicer3D(struct_arr).show()

arr = [[10, 20, 50, 54, 45], [10000, 10389945, 34285743, 438753495, 3489578943]]
arr1 = pd.DataFrame(arr[0])

plt.plot(arr1)
plt.show()