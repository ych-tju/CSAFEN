# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import torch

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def make_model(args, parent=False):
    return TDPN_SSFIB_HESSIAN()


class TDPN_SSFIB_HESSIAN(nn.Module):
    def __init__(self):
        super(TDPN_SSFIB_HESSIAN, self).__init__()


    def forward(self, x):
        B, C, H, W = x.shape
        sr = F.interpolate(x, size=[4*H, 4*W], mode='bicubic')

        return sr







x1 = torch.ones(16, 3, 127, 127)
# x2 = torch.ones(16, 64, 128, 128)
#
model = TDPN_SSFIB_HESSIAN()
y = model(x1)
print(y.shape)


# print(similarity.shape)
        # max_val, idx = torch.max(similarity, 1)
        # max_val = max_val.expand(B, self.n_class)
        #similarity = torch.where(similarity < max_val, zero, similarity)
        # print(similarity.shape)