from models.trihit_x import trihit_cth
from models.trihit_adapter_ft import trihit_cth_ft
from models.trihit_adapter_clip import trihit_cth_clip
from models.trihit_adapter_lora import trihit_cth_lora
from models.trihit_adapter_pit_v1 import trihit_cth_pit_v1
from models.trihit_adapter_pit_v2 import trihit_cth_pit_v2
from models.trihit_adapter_pit_v3 import trihit_cth_pit_v3
from models.trihit_adapter_pit_v4 import trihit_cth_pit_v4


models = {
    'trihit_cth': trihit_cth,
    'trihit_cth_ft': trihit_cth_ft,
    'trihit_cth_clip': trihit_cth_clip,
    'trihit_cth_lora': trihit_cth_lora,
    'trihit_cth_pit_v1': trihit_cth_pit_v1,
    'trihit_cth_pit_v2': trihit_cth_pit_v2,
    'trihit_cth_pit_v3': trihit_cth_pit_v3,
    'trihit_cth_pit_v4': trihit_cth_pit_v4
}

def model_builder(model_name, num_classes, args):
    model = models[model_name](num_classes=num_classes, dp_rate=args.dp_rate)
    return model
