"""Combine independently accumulated point-field normalization statistics."""

import argparse
import json
import math
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    stats = []
    for path in args.inputs:
        with open(path) as file:
            stats.append((path, json.load(file)))

    count = sum(item['count'] for _, item in stats)
    mean = sum(item['count'] * item['mean'] for _, item in stats) / count
    second_moment = sum(
        item['count'] * (item['std'] ** 2 + item['mean'] ** 2)
        for _, item in stats
    ) / count
    first = stats[0][1]
    combined = {
        'mean': mean,
        'std': math.sqrt(max(second_moment - mean ** 2, 0.0)),
        'count': count,
        'resolution': first['resolution'],
        'field_name': first.get('field_name', 'density'),
        'samples_per_mesh': first['samples_per_mesh'],
        'meshes_total': sum(item['meshes_total'] for _, item in stats),
        'meshes_valid': sum(item['meshes_valid'] for _, item in stats),
        'mesh_normalization': first['mesh_normalization'],
        'field_formula': first.get('field_formula', first.get('density_formula')),
        'sources': [path for path, _ in stats],
        'errors': [error for _, item in stats for error in item.get('errors', [])],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as file:
        json.dump(combined, file, indent=2)
    print(json.dumps({key: value for key, value in combined.items() if key != 'errors'}, indent=2))


if __name__ == '__main__':
    main()
