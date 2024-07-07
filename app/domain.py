import yaml
from dotmap import DotMap


def get_yaml(file):
    with open(file) as f:
        secret_dict = yaml.load(f, yaml.Loader)
    return DotMap(secret_dict)


secrets = get_yaml('secrets.yaml')
configs = get_yaml('configs.yaml')
