import torch
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

if __name__ == '__main__':
    xx()