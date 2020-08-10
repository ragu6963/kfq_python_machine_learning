# %%
# numpy 기본
import numpy as np

# %%
array1 = np.array([1, 2, 3])
print("type", type(array1))
print("형태", array1.shape)
print(array1.ndim)


# %%
array2 = np.array([[1, 2, 3], [4, 5, 6]])
print(type(array2))
print(array2.shape)
print(array2.ndim)

# %%
list1 = [1, 2, 3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)
# %%
list2 = [1, 2, "test"]
array1 = np.array(list2)
print(array1, array1.dtype)

list3 = [1, 2, 3.0]
array1 = np.array(list3)
print(array1, array1.dtype)
# %%
arr_int = np.array([1, 2, 3])
arr_float = arr_int.astype("float64")
print(arr_float, arr_float.dtype)

arr_float = arr_int.astype("int32")
print(arr_float, arr_float.dtype)

arr_float = np.array([1.1, 2.2, 3.3])
arr_int = arr_float.astype("int32")
print(arr_int, arr_int.dtype)

# %%
seq_arr = np.arange(10)
print(seq_arr)
print(seq_arr.dtype, seq_arr.shape)
# %%
zero_arr = np.zeros((3, 2), dtype="int32")
print(zero_arr)
print(zero_arr.dtype, zero_arr.shape)

one_arr = np.ones((3, 2))
print(one_arr)
print(one_arr.dtype, one_arr.shape)
# %%
arr1 = np.arange(10)
print(arr1)

arr2 = arr1.reshape(2, 5)
print(arr2)

arr3 = arr1.reshape(5, 2)
print(arr3)
# %%
# arr4 = arr1.reshape(4, 2)
arr1 = np.arange(10)
print(arr1)

arr2 = arr1.reshape(-1, 5)
print(arr2)

arr3 = arr1.reshape(3, -1)
print(arr3)

# %%
arr1 = np.arange(8)
arr3d = arr1.reshape(2, 2, 2)
print(arr3d)
print(arr3d.tolist())
print(type(arr3d.tolist()))
arr1d = arr3d.reshape(-1, 1)
print(arr1d)
print(arr1d.shape)


# %%
arr1 = np.arange(start=1, stop=10)
print(arr1)

value = arr1[2]
print(value)

# %%
arr1 = np.arange(start=1, stop=10)
arr3 = arr1[0:3]
print(arr3)
print(type(arr3))


# %%
arr1 = np.arange(start=1, stop=10)
arr4 = arr1[:3]
print(arr4)
arr5 = arr1[3:]
arr6 = arr1[:]
print(arr5)
print(arr6)

# %%
arr1 = np.arange(start=1, stop=10)
arr2d = arr1.reshape(3, 3)
print(arr2d[:2, :2])
print(arr2d[1:3, 0:3])

# %%
arr1 = np.arange(start=1, stop=10)
arr1[arr1 > 5]

# %%
arr1 > 5

# %%
org_arr = np.array([3, 1, 4, 1, 2, 2, 34, 78, 8])
print(org_arr.max())
print(np.max(org_arr))
print(np.argmax(org_arr))
# %%
org_arr = np.array([3, 1, 4, 1, 2, 2, 34, 78, 8])
print(np.sort(org_arr))
print(org_arr.sort())
print(org_arr)

# %%
org_arr = np.array([3, 1, 4, 1, 2, 2, 34, 78, 8])
print(np.sort(org_arr[::-1]))

# %%
arr2d = np.array([[8, 12], [1, 2]])
print(arr2d)

sort_axis0 = np.sort(arr2d, axis=0)
print(sort_axis0)

sort_axis1 = np.sort(arr2d, axis=1)
print(sort_axis1)


# %%
arr2d = np.array([[8, 12], [1, 2]])
np.argmax(arr2d, axis=0)
# %%
arr2d = np.array([[8, 12], [1, 2]])
np.argmax(arr2d, axis=1)

# %%
a = np.array([[1, 2, 3,], [3, 4, 5]])
b = np.array([[1, 2,], [3, 4], [5, 6]])
print(np.dot(a, b))


# %%
a = [[1, 2], [3, 4]]
print(np.transpose(a))

# %%
a = np.array([[1, 2], [3, 4]])
b = 5
a + b


# %%
c = np.array([[1, 2], [3, 4]])
d = np.array([10, 20])
c + d


# %%
c = np.array([[1, 2], [3, 4]])
d = np.array([[10], [20]])
c + d


# %%
a = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
iter = np.nditer(a, flags=["multi_index"], op_flags=["readwrite"])

while not iter.finished:
    idx = iter.multi_index
    print(a[idx])
    iter.iternext()
# %%
a = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
type(a)
print(a.shape)
row = np.array([1, 2, 3, 4]).reshape(1, 4)
col = np.array([1, 2]).reshape(2, 1)
b = np.concatenate((a, row), axis=0)
c = np.concatenate((a, col), axis=1)
print(b)
print(c)

# %%
