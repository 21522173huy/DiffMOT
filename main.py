from diffmot import DiffMOT
import argparse
import yaml
from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='', help='Path to the config file')
    parser.add_argument('--dataset', default='', help='Dataset name')
    parser.add_argument('--network', choices=['ReUNet', 'ReUNet+++', 'Smaller'], help='Unet version')
    parser.add_argument('--data_dir', default=None, help='Path to the data directory')
    parser.add_argument('--det_dir', default=None, help='Path to the data directory')
    parser.add_argument('--info_dir', default=None, help='Path to the data directory')
    parser.add_argument('--reid_dir', default=None, help='Path to the data directory')
    parser.add_argument('--save_dir', default=None, help='Path to the data directory')
    parser.add_argument('--checkpoint_path', default=None, help='Path to the data directory')

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

    agent = DiffMOT(config)

    if config.eval_mode:
        agent.eval()
    else:
        agent.train()

if __name__ == '__main__':
    main()
