import torch
import torch.nn as nn
x = torch.FloatTensor(torch.rand([16, 100, 128, 9]))
y = torch.FloatTensor(torch.rand([64, 9]))


y = y.expand(16, 100, 128, 64, 9).transpose(2, 3).transpose(1, 2).transpose(0, 1)
x = x.expand(64, 16, 100, 128, 9)
print(y.shape)
print(x.shape)
similarity = torch.cosine_similarity(x, y, dim=4).transpose(0, 1).transpose(1, 2).transpose(2, 3)


sigmoid = nn.Sigmoid()
similarity = sigmoid(similarity)
print(similarity.shape)
x = torch.matmul(similarity, y)
print(x.shape)