import copy
import yaml

# deepcopy src values into target(Union(target, src) but src primary)
def inherit_dict(target, src):
    ret = copy.deepcopy(target)

    for k, v in src.items():
        # hierachical dictionary
        if isinstance(v, dict):
            ret[k] = inherit_dict(ret.get(k,dict()), v)

        # value(or list)
        else:
            ret[k] = copy.deepcopy(v)

    return ret

# Load Configuration yaml file
# cfg_path = os.path.join(args.config_dir, 'finetune', args.network+'.yml')
def load_config(cfg_path):

    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Inherit base configuration
    if 'base' in cfg:
        cfg_parent = load_config(cfg['base'])
        cfg = inherit_dict(cfg_parent, cfg)
    
    return cfg