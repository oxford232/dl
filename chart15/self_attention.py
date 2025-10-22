import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(3)

N = 3
D = 4


all_x = []
for n in range(N):
    all_x.append(np.random.normal(size=(D,1)))

all_x = torch.tensor(all_x).squeeze()


np.random.seed(0)

omega_q = torch.tensor(np.random.normal(size=(D,D)))
omega_k = torch.tensor(np.random.normal(size=(D,D)))
omega_v = torch.tensor(np.random.normal(size=(D,D)))
beta_q = torch.tensor(np.random.normal(size=(D)))
beta_k = torch.tensor(np.random.normal(size=(D)))
beta_v = torch.tensor(np.random.normal(size=(D)))



Q = all_x @ omega_q + beta_q
K = all_x @ omega_k + beta_k
V = all_x @ omega_v + beta_v

# print(Q.shape, Q)
# Q = Q.view(N, 2, 2)
# print(Q.shape, Q)




# scores = torch.matmul(all_queries, all_keys.T)

# d_k = all_queries.size(-1)
# scores = scores / d_k**0.5

# attn_weights = F.softmax(scores, dim=1)

# print(attn_weights[0][0] + attn_weights[0][1] + attn_weights[0][2])

scores = Q @ K.T
attn_weights = F.softmax(scores, dim=1)

print(attn_weights[0][0] + attn_weights[0][1] + attn_weights[0][2])


# print(attn_weights @ all_values)

    
    