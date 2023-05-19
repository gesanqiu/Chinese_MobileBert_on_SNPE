import numpy as np


data = [101, 1963, 862, 3291, 2940, 5709, 1446, 5308, 2137, 7213, 6121, 1305, 102]
array = np.array(data)
# 保存数组到文件中
array.tofile(r'.\data\test_13.raw')