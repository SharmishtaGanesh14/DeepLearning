import torch
import numpy as np

# Create a simple Python list of lists (3x3 matrix)
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Convert Python list to a torch tensor
x_data = torch.tensor(data)

# Convert Python list to a NumPy array
np_array = np.array(data)

# Convert NumPy array to a torch tensor (shares memory!)
x_np = torch.from_numpy(np_array)

# Create a tensor of ones with the same shape and dtype as x_data
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

# Create a tensor with the same shape as x_data, but with random float values
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# Define a shape for future tensors
shape = (2, 3)

# Create tensors of the given shape with different initializations
rand_tensor = torch.rand(shape)      # Random values from [0, 1)
ones_tensor = torch.ones(shape)      # All values 1
zeros_tensor = torch.zeros(shape)    # All values 0

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Create a 3x4 tensor with random values
tensor = torch.rand(3, 4)

# Inspect its shape, data type, and device location (CPU/GPU)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Move tensor to accelerator (GPU) if available (NOTE: .accelerator is incorrect, fixed below)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Create a 4x4 tensor of ones
tensor = torch.ones(4, 4)

# Access specific rows and columns
print(f"First row: {tensor[0]}")       # Row 0
print(f"First column: {tensor[:, 0]}") # All rows, column 0

# Set the second column (index 1) to zero
tensor[:, 1] = 0
print(f"Second column after zeroing: {tensor[:, 1]}")
print(tensor)

# Concatenate the tensor 3 times horizontally (column-wise)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Matrix multiplication in three ways (all should produce same result)
y1 = tensor.T @ tensor
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)  # Save result into y3

# Element-wise multiplication (Hadamard product) in three ways
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)  # Save result into z3

# Aggregate all elements of tensor into a scalar (sum)
agg = tensor.sum()

# Extract Python scalar value from tensor
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operation: add 5 to all elements (modifies the tensor directly)
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Demonstrating shared memory between torch tensor and numpy array

t = torch.ones(5)
print(f"t: {t}")  # Tensor of 1s

n = t.numpy()     # NumPy array shares memory with `t`
print(f"n: {n}")  # Same values (1s)

# In-place add 1 to tensor `t` → also affects `n` since they share memory
t.add_(1)
print(f"t: {t}")  # Now 2s
print(f"n: {n}")  # Now 2s

# Now starting from NumPy → Tensor (again shares memory)
n = np.ones(5)
t = torch.from_numpy(n)

# Modify numpy array in-place → reflects in torch tensor
np.add(n, 1, out=n)
print(f"t: {t}")  # Now 2s
print(f"n: {n}")  # Now 2s
