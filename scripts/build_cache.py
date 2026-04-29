import torch
from lead.data_loader.carla_dataset import CARLAData
from lead.training.config_training import TrainingConfig
from tqdm import tqdm

# Restrict cache building to this key list.
# Keys follow CacheKey.__str__ format, e.g. scenario_route_frame_False)
key_list: list[str] = [
    # "DynamicObjectCrossing_Town04_Rep0_Town04_Scenario3_13_route0_04_25_06_03_14_0004_True)",
]

config = TrainingConfig()
config.use_persistent_cache = True
config.use_training_session_cache = False
config.force_rebuild_data_cache = True
config.cache_key_filter_list = key_list or None

print(
    "cache_key_filter_list",
    0 if config.cache_key_filter_list is None else len(config.cache_key_filter_list),
)
if config.cache_key_filter_list:
    print("cache_key_filter_list_examples", config.cache_key_filter_list[:5])

for k, v in config.training_dict().items():
    print(k, v)

data = CARLAData(
    root=config.carla_data,
    config=config,
    training_session_cache=None,
    build_cache=True,
)
dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=config.assigned_cpu_cores,
    shuffle=False,
    num_workers=config.assigned_cpu_cores,
    prefetch_factor=1,
)

for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
    pass
