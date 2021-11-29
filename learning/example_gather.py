import torch
import numpy as np
'''
    input.shape = (q,3)
    idx.shape = (1,3) = out.shape
    input 与 index 只有一个维度不同,在这个不同的维度上做idx
'''
def xx():
    q_table = torch.arange(3, 12).view(3, 3)
    print(q_table)
    r'''
        按行取
        out = [input[idx[0,0],0],input[idx[0,1],1],input[idx[0,2],2]]
    '''
    index = torch.tensor([2, 1, 0,2,1,1,2])
    tensor_0 = q_table.gather(0,index.unsqueeze(-1).expand((index.shape[0],q_table.shape[1])).long())
    print(tensor_0)
    r'''
        按列取
        out = [input[0,idx[0,0]],input[1,idx[1,0]],input[2,idx[2,0]]]
    '''
    index = torch.tensor([2, 1,0])
    tensor_1 = q_table.gather(1, index.unsqueeze(-1)).squeeze(-1).long()
    print(tensor_1)
    r'''
        按行取0列 特殊用法
        out = [input[idx[0,0],0],input[idx[1,0],0],input[idx[2,0],0]]
    '''
    index = torch.tensor([2, 1, 0])
    tensor_1 = q_table.gather(0, index.unsqueeze(-1)).squeeze(-1).long()
    print(tensor_1)

def step_obs():
    q_table = torch.arange(60).view(3,4, 5)
    idx = torch.tensor([2,1],dtype=torch.long)
    r"""
        q_table[2,1] == tensor([45, 46, 47, 48, 49])
        用q_table.gather(idx)怎么做???
    """
    print()
def np_array_idx():
    ll = np.array([[[1, 1, 9,2,3], [2, 2, 8,2,3]],[[1, 2, 9,4,3], [2, 3, 8,4,3]],[[0, 1, 9,2,6], [1, 2, 8,2,6]]])
    mapp = np.arange(72*2).reshape((3,4,6,2))
    r'''
    arr维度为mapp没有被index的维度+(i1,i2,,,in)
    arr[i1,i2,,,in] = mapp[idx1[i1,i2,,,in],idx2[i1,i2,,,in],idx3[i1,i2,,,in],idx4[i1,i2,,,in],,,idxn[i1,i2,,,in]]
    index之外的维度可能需要broadcast过滤
        broadcast dim0 (3,)->(3,1)->(3,2)         
                                    idx1        idx2'''
    mapp[np.array([0, 1, 2])[:, None], ll[:, :, 0], ll[:, :, 1]] = ll[...,[2,4]]
    print()
if __name__ == '__main__':
    np_array_idx()