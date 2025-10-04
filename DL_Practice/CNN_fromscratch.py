import torch
import numpy as np

def apply_padding(image,padding):
    H,W=image.shape
    column=np.zeros((H,padding))
    image_1=np.hstack([column,image,column])
    W_new=image_1.shape[1]
    row=np.zeros((padding,W_new))
    image_2=np.vstack([row,image_1,row])
    return image_2

def convolution_op(image,filters,striding=1,padding=1):
    C, H, W = image.shape
    N, C_f, K, K_f = filters.shape
    assert C == C_f, "Number of channels in image and filters must match"

    image_padded=[]
    for i in range(C):
        image_padded.append(apply_padding(image[i],padding))

    image_padded=np.array(image_padded)
    _,H_p,W_p=image_padded.shape

    H_out=(H_p-K)//striding +1
    W_out=(W_p-K)//striding +1

    out=np.zeros((N,H_out,W_out))

    for i in range(0,N):
        for j in range(0,H_out*striding,striding):
            for k in range(0,W_out*striding,striding):
                patch=image_padded[:,j:j+K,k:k+K]
                out[i,j//striding,k//striding]=np.sum(patch*filters[i])

    return out

def max_pooling_op(image,filter_size,striding=1):
    C, H, W = image.shape

    H_out=(H-filter_size)//striding +1
    W_out=(W-filter_size)//striding +1

    out=np.zeros((C,H_out,W_out))
    for i in range(C):
        for j in range(0,H_out*striding,striding):
            for k in range(0,W_out*striding,striding):
                patch=image[i,j:j+filter_size,k:k+filter_size]
                out[i,j//striding,k//striding]=np.max(patch)
    return out

def batch_convolution(images,filters,striding,padding):
    batch_size=images.shape[0]
    batch_out=[]
    for i in range(batch_size):
        batch_out.append(convolution_op(images[i],filters,striding,padding))
    return np.array(batch_out)

def batch_max_pooling(images,filter_size,striding):
    batch_size=images.shape[0]
    batch_out=[]
    for i in range(batch_size):
        batch_out.append(max_pooling_op(images[i],filter_size,striding))
    return np.array(batch_out)

batchsize=10
images=torch.randn(batchsize, 3, 64, 64).numpy()
filters=torch.randn(32,3,3,3).numpy()
conv_out=batch_convolution(images,filters,1,1)
print(conv_out.shape)
max_pool_out=batch_max_pooling(conv_out,3,1)
print(max_pool_out.shape)




