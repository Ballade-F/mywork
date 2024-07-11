import torch
import torch.nn as nn
import torch.nn.functional as F

batch = 4
nodeSize = 5
embeddingSize = 2

mask = torch.zeros(batch, nodeSize,dtype=torch.bool)
print(mask)
# idx = torch.zeros(batch,dtype=torch.long)
idx_last = torch.zeros(batch,dtype=torch.long)

idx = torch.Tensor([0, 4, 2, 3]).long()
mask[torch.arange(batch), idx] = 1
print(torch.arange(batch))
print(mask)


p = torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.1, 0.2, 0.3, 0.4, 0.5]])
print(torch.multinomial(p, 1)[:,0].shape)

pro = torch.FloatTensor(batch, nodeSize)
print(p[torch.arange(batch), idx].shape)
pro[:,0] = p[torch.arange(batch), idx]

dis = torch.rand(batch, nodeSize, nodeSize,1)
# print(dis)
print(dis[torch.arange(batch), idx_last, idx].squeeze())