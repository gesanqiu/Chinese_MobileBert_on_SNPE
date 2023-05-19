import numpy as np

float_array = np.fromfile(r".\result\result_13_op11_Add_221.raw", dtype=np.float32)
print(float_array)
array = float_array.reshape((1, 64, 768))
np.savetxt(r".\result\result_13_op11_Add_221.txt", array.squeeze())
# weights = np.load(r".\data\mobilebert_gather54_weights.npy")
# print(weights.shape)
# print(weights)
# weights.tofile("mobilebert_gather54_weights.bin")
# np.savetxt(r".\data\mobilebert_gather54_weights.txt", weights.squeeze())