import wandb

# 定义超参数搜索空间
sweep_config = {
    'method': 'bayes',  # 使用贝叶斯优化
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'   
    },
    'parameters': {
        # fixed parameters
        'window': {
            'value': 80
        },
        'norm_type': {
            'value': 'total'
        },
        'require_norm_data': {
            'value': True
        },
        'input_dim': {
            'value': 10
        },
        'output_dim': {
            'value': 6
        },
        'epochs': {
            'value': 50
        },
        # tuned parameters
        'gru_hidden_size': {
            'values': [32, 64, 128, 256]
        },
        'gru_layers': {
            'values': [1, 2, 3]
        },
        'tcn_channels': {
            'values': [32, 64, 128, 256]
        },
        'tcn_layers': {
            'values': [1, 2, 3]
        },
        'alpha': {
            'values': [0.7]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,   # 直接指定最小值
            'max': 1e-3,   # 直接指定最大值
        },
        'batch_size': {
            'values': [512, 1024, 2048]
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,   # 直接指定最小值
            'max': 1e-3,   # 直接指定最大值
        },
        'position_weight': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 2.0
        },
        'orientation_weight': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 2.0
        },
        'require_norm_data': {
            'value': True
        }
    }
}

# 定义wandb项目配置
wandb_config = {
    'project': 'pelican',
    'entity': '13610678931',  # 你的wandb用户名
    'name': 'dronepose-sweep',
    'notes': 'Hyperparameter optimization for pelican using pelican model'
}