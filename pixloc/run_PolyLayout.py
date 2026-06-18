import argparse
import json
import time
from pathlib import Path

import torch
import tqdm
from omegaconf import OmegaConf

from .pixlib.datasets import get_dataset
from .pixlib.geometry import Cuboid
from .pixlib.utils.experiments import load_experiment
from .pixlib.utils.tensor import batch_to_device
from .pixlib.utils.tools import set_seed


def main(conf, experiment: str, split: str, flatten: bool, output_path: Path) -> None:
    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader(split, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_experiment(experiment, conf.model).to(device)

    start_time = time.time()
    layout_preds = []

    for data in tqdm.tqdm(loader):
        data = batch_to_device(data, device, non_blocking=True)

        with torch.no_grad():
            pred = model(data)

        room_preds = {}
        for i, layout in enumerate(pred['layout_opt'][-1]):
            attr = ('R', 't', 's') if isinstance(layout, Cuboid) else ('verts', 'faces')
            pred = {a: getattr(layout, a).cpu().numpy().tolist() for a in attr}
            if model.conf.optimizer.name == 'multi_room_optimizer' and not flatten:
                room_preds[data['room'][i]] = pred
            else:
                layout_preds.append(pred)
        if room_preds:
            layout_preds.append(room_preds)

        del pred, data
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    num_pred = len(loader.dataset)
    print(f'Predicted {num_pred} layouts in {elapsed:.2f} s.')
    print(f'Average time per prediction: {1000.0*elapsed/num_pred:.2f} ms')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(layout_preds, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PolyLayout')
    parser.add_argument('--experiment', '-exp', required=True, help='Experiment to use')
    parser.add_argument('--conf', '-c', type=Path, required=True, help='Path to config file')
    parser.add_argument('--split', '-s', required=True, help='Data split (e.g., train, val, test)')
    parser.add_argument('--flatten', '-f', action='store_true', help='Flatten multi-room predictions')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Path to output JSON prediction file')
    parser.add_argument('extra_args', nargs='*')
    parser.add_argument('--seed', '-seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    conf = OmegaConf.load(args.conf)
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.extra_args))
    main(conf, args.experiment, args.split, args.flatten, args.output)
