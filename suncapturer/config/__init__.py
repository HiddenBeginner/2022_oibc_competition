config = {
    'seed': 0,
    'k': 0.5,
    'epochs': 500,
    'hidden_dim': 10,
    'last_data': '2022-11-15',

    'ids': range(0, 21),

    'features': ['temperature', 'humidity', 'dew_point', 'wind_dir', 'wind_spd',
                 'uv_idx', 'visibility', 'ceiling', 'cloudiness', 'precip_prob',
                 'precip_1h', 'forecast', 'capacity', 'hour', 'year', 'month', 'day'],

    'optimizer': {
        'lr': 0.0,
        'weight_decay': 0.01
    },

    'scheduler': {
        'T_0': 500,
        'T_mult': 1,
        'eta_max': 0.001,
        'T_up': 10,
        'gamma': 0.5,
    },

    'loss': {
        'loss': 'MeanStdLoss',
        'k': 1.0,
        'weights': (1.0, 0.0, 0.0),
    },

    'wandb': {
        'project': 'solar_power_prediction',
        'name': 'WaveNet'
    }
}
