import yaml
# import ipdb

def load_model_config(model, mode):
    config_path = f'config/{model}.yaml'
    print(f'[!] load configuration from {config_path}')
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
        new_config = {}
        for key, value in configuration.items():
            if key in ['train', 'train_asyn', 'train_pipeline', 'test', 'inference', 'pretrain', 'queryside', 'baseline']:
                if mode == key:
                    new_config.update(value)
            else:
                new_config[key] = value
        configuration = new_config
    return configuration

def load_config(args):
    '''the configuration of each model can rewrite the base configuration'''
    # base config
    base_configuration = load_base_config()

    # load one model config
    configuration = load_model_config(args['model'], args['mode'])

    # update and append the special config for base config
    base_configuration.update(configuration)
    configuration = base_configuration

    # load by lang
    args['lang'] = configuration['datasets'][args['dataset']]
    return configuration

def load_base_config():
    config_path = f'config/base.yaml'
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    print(f'[!] load base configuration: {config_path}')
    return configuration
