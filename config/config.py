# Configuration for the project

# Features to keep for DDoS detection
FEATURES_TO_KEEP = [
    'sload',
    'spkts',
    'sinpkt',
    'ct_dst_ltm',
    'ct_srv_dst',
    'dur',
    'sbytes',
    'dbytes',
    'state',
    'proto',
    'dpkts',
    'service',
    'smean',
    'ct_src_ltm',
    'sloss',
    'dloss',
    'synack', # important
    'tcprtt'
]

# Categorical features to be one-hot encoded
CATEGORICAL_FEATURES = ['state', 'proto', 'service']
