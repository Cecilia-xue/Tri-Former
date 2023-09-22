import torch
from models.trihit_adapter_pit_v1 import trihit_cth_pit_v1
from models.trihit_adapter_pit_v2 import trihit_cth_pit_v2
from models.trihit_adapter_pit_v3 import trihit_cth_pit_v3

input = torch.randn(4, 1, 200, 27, 27).cuda()
model = trihit_cth_pit_v3().cuda()
output = model(input)
print('done')

