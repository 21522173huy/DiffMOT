
from diffmot import DiffMOT
import argparse
import yaml
from easydict import EasyDict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='', help='Path to the config file')
    parser.add_argument('--dataset', default='', help='Dataset name')
    parser.add_argument('--network', choices=['ReUNet', 'ReUNet+++', 'Smaller'], help='Unet version')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the network')
    parser.add_argument('--filters', type=int, nargs='+', help="List of filters")
    parser.add_argument('--skip_connection', type=str2bool, default=False, help='Skp connection')
    parser.add_argument('--data_dir', default=None, help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--early_stopping', choices=['loss', 'iou'])
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       if v is not None:
           config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset
    config = EasyDict(config)
    
    # Update config with command-line arguments if provided
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.network is not None:
        config.network = args.network
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.filters is not None:
        config.filters = args.filters
    if args.skip_connection is not None:
        config.skip_connection = args.skip_connection
    config.early_stopping = args.early_stopping

    agent = DiffMOT(config)

    if config.eval_mode:
        agent.eval()
    else:
        agent.train()

if __name__ == '__main__':
    main()
