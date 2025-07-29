# Check actual parameter counts
from src.model_architecture import get_morris_model_configs
configs = get_morris_model_configs()
for name, config in configs.items():
    print(f"{name}: {config['parameters']['total_params']:,} parameters")
