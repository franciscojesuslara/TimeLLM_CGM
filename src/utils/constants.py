from pathlib import Path
from ray import tune
import matplotlib.pyplot as plt
PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROJECT_REPORTS = Path.joinpath(PATH_PROJECT_DIR, 'reports')
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'modelos')
seeds=[0, 45, 34, 15]
timeseries_length=1452
out_ts_length=12
bbd_name='vivli'
max_epochs=15
samples_day=288
seq_len= 96

config_lstm = {
  "input_size": tune.choice([seq_len]),
  "h": None,
  "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
  "encoder_n_layers": tune.randint(1, 4),
  "context_size": tune.choice([5, 10, 50]),
  "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
  "learning_rate": tune.loguniform(1e-4, 1e-2),
  "scaler_type": tune.choice(['robust', 'minmax', 'standard']),
  "batch_size": tune.choice([3, 6, 10]), 
  "random_seed": tune.randint(1, 5),
  "max_steps": tune.choice([500]),
  "val_check_steps": tune.choice([10])
}
config_tcn = {
  "input_size": tune.choice([seq_len]),
  "h": None,
  "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
  "context_size": tune.choice([5, 10, 50]),
  "decoder_hidden_size": tune.choice([64, 128, 256]),
  "learning_rate": tune.loguniform(1e-4, 1e-1),
  "scaler_type": tune.choice(['robust', 'minmax', 'standard']),
  "batch_size": tune.choice([3, 6, 10]), 
  "random_seed": tune.randint(1, 5),
  "max_steps": tune.choice([500]),
  "val_check_steps": tune.choice([10])
}

config_nhits = {
  "input_size": tune.choice([seq_len]),
  "h": None,
  "n_pool_kernel_size": tune.choice(
    [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
  ),
  "n_freq_downsample": tune.choice(
    [
      [168, 24, 1],
      [24, 12, 1],
      [180, 60, 1],
      [60, 8, 1],
      [40, 20, 1],
      [1, 1, 1],
    ]
  ),
  "learning_rate": tune.loguniform(1e-4, 1e-1),
  "scaler_type": tune.choice(['robust', 'minmax', 'standard']),
  "batch_size": tune.choice([3, 6, 10]), 
  "windows_batch_size": tune.choice([32, 64, 128]), 
  "random_seed": tune.randint(1, 5),
  "max_steps": tune.choice([1000]),
  "val_check_steps": tune.choice([10])
}
config_tide = {
  "input_size": tune.choice([seq_len]),
  "h": None,
  "hidden_size": tune.choice([256, 512, 1024]),
  "decoder_output_dim": tune.choice([8, 16, 32]),
  "temporal_decoder_dim": tune.choice([32, 64, 128]),
  "num_encoder_layers": tune.choice([1, 2, 3]),
  "num_decoder_layers": tune.choice([1, 2, 3]),
  "temporal_width": tune.choice([4, 8, 16]),
  "dropout": tune.choice([0.0, 0.1, 0.2, 0.3, 0.5]),
  "layernorm": tune.choice([True, False]),
  "scaler_type": tune.choice(['robust', 'minmax', 'standard']),
  "batch_size": tune.choice([3, 6, 10]), 
  "learning_rate": tune.loguniform(1e-4, 1e-2), 
  "random_seed": tune.randint(1, 5),
  "max_steps": tune.choice([1000]),
  "val_check_steps": tune.choice([10])
  }

config_patchtst = {
  "input_size": tune.choice([seq_len]),
  "h": None,
  "encoder_layers": tune.choice([2, 4, 8]),
  "hidden_size": tune.choice([16, 128, 256]),
  "n_heads": tune.choice([4, 16, 32]),
  "patch_len": tune.choice([16, 24, 36]),
  "learning_rate": tune.loguniform(1e-4, 1e-1),
  "scaler_type": tune.choice(['robust', 'minmax', 'standard']),
  "batch_size": tune.choice([3, 6, 10]),
  "random_seed": tune.randint(1, 5),
  "max_steps": tune.choice([1000]),
  "val_check_steps": tune.choice([10])
  }


config_tsmixer = {
  "input_size": tune.choice([seq_len]),
  "h": None,
  "n_series": None,
  "n_block": tune.choice([1, 2, 4, 6, 8]),
  "dropout": tune.choice([0.3, 0.6, 0.9]),
  "ff_dim": tune.choice([32, 64, 128]),
  "scaler_type": tune.choice(['robust', 'minmax', 'standard']),
  "batch_size": tune.choice([3, 6, 10]), 
  "learning_rate": tune.loguniform(1e-4, 1e-1),
  "random_seed": tune.randint(1, 5),
  "max_steps": tune.choice([1000]),
  "val_check_steps": tune.choice([10])
  }



