## installation pytorch
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


import torch

# check if cuda is available
print(torch.cuda.is_available())


# cuda version - dont know what cuda is ....yet
print(torch.version.cuda)


# using cuda
# lets have a tensor which will be explained later,

# on cpu
t = torch.tensor([1, 2, 3, 5])
print(t)


# on gpu
t = t.cuda()

print(t)


# so tensor are multiple dimensional tensor
