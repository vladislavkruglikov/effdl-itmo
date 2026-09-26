import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy
import scipy.optimize
import torch

matplotlib.use('Agg')
import matplotlib.pyplot

from equations import bytes_moved, energy, flops, latency, memory
from measure import write_readme_section
from models import CNN


PLOT_UNITS = {'latency': 'seconds', 'memory': 'bytes', 'energy': 'joules'}
SPLITS = [('training', False, 'o'), ('validation', True, 'x')]


def load_measurements(path):
    with Path(path).open(newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('Measurements file is empty')

    data = {}
    for name in ['S', 'B', 'latency', 'memory', 'energy', 'free_bytes_before']:
        data[name] = numpy.array([float(row[name]) if row.get(name) else numpy.nan for row in rows])
    flags = [row['is_validation'].strip().lower() for row in rows]
    if any(flag not in ('0', '1', 'false', 'true') for flag in flags):
        raise ValueError('is_validation must contain 0/1 or false/true')
    data['is_validation'] = numpy.array([flag in ('1', 'true') for flag in flags])
    data['status'] = numpy.array([row['status'].upper() for row in rows])
    if not numpy.isfinite(data['S']).all() or not numpy.isfinite(data['B']).all():
        raise ValueError('Image sizes and batches must be finite numbers')
    if len(set(zip(data['S'], data['B']))) != len(rows):
        raise ValueError('Duplicate image-size/batch configurations in measurements')
    # Recompute predictions from the current equations rather than trust old CSV estimates
    data['predicted_memory'] = memory(data['S'], data['B'])
    return data


def valid_points(data, quantity):
    values = data[quantity]
    return (data['status'] == 'OK') & numpy.isfinite(values) & (values > 0)


def fit_latency(data):
    training = valid_points(data, 'latency') & ~data['is_validation']
    if training.sum() < 4:
        raise ValueError('Need at least four valid training latency measurements')
    sizes = data['S'][training]
    batches = data['B'][training]
    measured = data['latency'][training]

    def parameters(log_values):
        overhead, compute_speed, bandwidth = numpy.exp(log_values)
        return {
            'launch_seconds': float(overhead),
            'compute_flops_per_second': float(compute_speed),
            'bandwidth_bytes_per_second': float(bandwidth),
        }

    def residuals(log_values):
        predicted = latency(sizes, batches, parameters(log_values))
        # Log errors give small and large configurations comparable influence
        return numpy.log(predicted / measured)

    starts = [(1e-5, 1e12, 1e11), (1e-6, 1e13, 1e12), (1e-4, 1e11, 1e10)]
    bounds = (numpy.log([1e-9, 1e7, 1e6]), numpy.log([1, 1e16, 1e15]))
    best = None
    for start in starts:
        result = scipy.optimize.least_squares(residuals, numpy.log(start), bounds=bounds)
        if result.success and numpy.isfinite(result.cost):
            if best is None or result.cost < best.cost:
                best = result
    if best is None:
        raise RuntimeError('Latency fitting did not converge')
    return parameters(best.x)


def fit_energy(data, latency_theta):
    training = valid_points(data, 'energy') & ~data['is_validation']
    if training.sum() < 4:
        return None
    sizes = data['S'][training]
    batches = data['B'][training]
    measured = data['energy'][training]

    # Columns are predicted seconds, arithmetic operations and transferred bytes
    features = numpy.column_stack([
        latency(sizes, batches, latency_theta),
        flops(sizes, batches),
        bytes_moved(sizes, batches),
    ])
    # Divide by measured energy to fit relative errors, then normalize column scales
    relative_features = features / measured[:, None]
    scales = numpy.linalg.norm(relative_features, axis=0)
    coefficients, _ = scipy.optimize.nnls(relative_features / scales, numpy.ones_like(measured))
    watts, joules_per_flop, joules_per_byte = coefficients / scales
    return {
        'watts': float(watts),
        'joules_per_flop': float(joules_per_flop),
        'joules_per_byte': float(joules_per_byte),
        'latency': latency_theta,
    }


def predict(data, theta):
    sizes, batches = data['S'], data['B']
    predictions = {
        'latency': latency(sizes, batches, theta['latency']),
        'memory': memory(sizes, batches),
    }
    if theta['energy'] is not None:
        predictions['energy'] = energy(sizes, batches, theta['energy'])
    return predictions


def metrics(measured, predicted):
    if len(measured) == 0:
        return {'n': 0}
    relative_error = numpy.abs(predicted / measured - 1)
    return {
        'n': len(measured),
        'mape_percent': float(100 * relative_error.mean()),
        'median_ape_percent': float(100 * numpy.median(relative_error)),
        'p90_ape_percent': float(100 * numpy.quantile(relative_error, .9)),
    }


def predicted_oom(data):
    available = data['free_bytes_before']
    return numpy.isfinite(available) & (data['predicted_memory'] > available)


def evaluate(data, predictions):
    report = {'counts': {status: int((data['status'] == status).sum()) for status in ['OK', 'OOM', 'ERROR']}}
    for quantity, predicted in predictions.items():
        report[quantity] = {}
        for label, validation, _ in SPLITS:
            selected = valid_points(data, quantity) & (data['is_validation'] == validation)
            report[quantity][label] = metrics(data[quantity][selected], predicted[selected])

    # The activation-sum estimate is the course convention, not a bound on the live peak
    predicted = predicted_oom(data)
    observed = data['status'] == 'OOM'
    known = numpy.isfinite(data['free_bytes_before']) & numpy.isin(data['status'], ['OK', 'OOM'])
    report['oom_comparison'] = {
        'predicted_oom': int((predicted & known).sum()),
        'observed_oom': int(observed.sum()),
        'missed_oom': int((observed & ~predicted & known).sum()),
        'false_positive': int((predicted & ~observed & known).sum()),
        'unassessed': int((~known).sum()),
    }
    return report


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    matplotlib.pyplot.close(figure)


def plot_parity(data, predicted, quantity, path):
    figure, axis = matplotlib.pyplot.subplots(figsize=(6, 5))
    valid = valid_points(data, quantity)
    if not valid.any():
        matplotlib.pyplot.close(figure)
        return
    for label, validation, marker in SPLITS:
        selected = valid & (data['is_validation'] == validation)
        axis.scatter(data[quantity][selected], predicted[selected], label=label, marker=marker, alpha=.7)
    values = numpy.concatenate([data[quantity][valid], predicted[valid]])
    low, high = values.min() * .9, values.max() * 1.1
    axis.plot([low, high], [low, high], 'k--', label='perfect prediction')
    unit = PLOT_UNITS[quantity]
    axis.set(xscale='log', yscale='log', xlabel=f'Measured {quantity} ({unit})',
             ylabel=f'Predicted {quantity} ({unit})')
    axis.legend()
    save_figure(figure, path)


def plot_grid(data, predicted, quantity, path):
    sizes = numpy.unique(data['S'])
    columns = min(4, len(sizes))
    rows = (len(sizes) + columns - 1) // columns
    figure, axes = matplotlib.pyplot.subplots(rows, columns, figsize=(4 * columns, 3.3 * rows), squeeze=False)
    for axis, size in zip(axes.flat, sizes):
        selected = data['S'] == size
        order = numpy.argsort(data['B'][selected])
        # Show the equation at all sampled batches, even when measurement failed
        axis.plot(data['B'][selected][order], predicted[selected][order],
                  color='black', label='Equation prediction', zorder=2)
        for label, validation, marker in SPLITS:
            points = selected & valid_points(data, quantity) & (data['is_validation'] == validation)
            if points.any():
                axis.scatter(data['B'][points], data[quantity][points], marker=marker,
                             color='tab:blue' if not validation else 'tab:orange',
                             label=f'T4 measured ({label})', zorder=3)
        axis.set(title=f'Image side S = {int(size)} pixels', xlabel='Batch size B (images)',
                 ylabel=f'{quantity.capitalize()} ({PLOT_UNITS[quantity]} per forward)',
                 xscale='log', yscale='log')
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(sizes):]:
        axis.set_visible(False)
    save_figure(figure, path)


def plot_oom(data, path):
    figure, axis = matplotlib.pyplot.subplots(figsize=(8, 5))
    for status, marker in [('OK', 'o'), ('OOM', 'x'), ('ERROR', 's')]:
        selected = data['status'] == status
        if selected.any():
            axis.scatter(data['S'][selected], data['B'][selected], marker=marker, label=status)
    predicted = predicted_oom(data)
    if predicted.any():
        axis.scatter(data['S'][predicted], data['B'][predicted], s=110, facecolors='none',
                     edgecolors='red', label='activation-sum estimate exceeds free')
    axis.set(xlabel='Image side S (pixels)', ylabel='Batch B', yscale='log')
    axis.set_title(f'Observed OOM: {int((data["status"] == "OOM").sum())}; predicted OOM: {int(predicted.sum())}')
    axis.legend()
    save_figure(figure, path)


def plot_regimes(data, theta, path):
    sizes = numpy.arange(int(data['S'].min()), int(data['S'].max()) + 1, 16)
    batches = numpy.arange(int(data['B'].min()), int(data['B'].max()) + 1)
    s, b = numpy.meshgrid(sizes, batches)

    def components(s, b):
        compute = flops(s, b) / theta['compute_flops_per_second']
        transfer = bytes_moved(s, b) / theta['bandwidth_bytes_per_second']
        overhead = numpy.full_like(compute, 17 * theta['launch_seconds'])
        return numpy.stack([overhead, transfer, compute])

    # Classify the largest model term, not a hardware-profiled bottleneck
    regimes = numpy.argmax(components(s, b), axis=0)
    measured_regimes = numpy.argmax(components(data['S'], data['B']), axis=0)
    counts = numpy.bincount(measured_regimes, minlength=3)
    figure, axis = matplotlib.pyplot.subplots(figsize=(9, 6))
    colors = matplotlib.colors.ListedColormap(['#dce7f7', '#f5ca80', '#9fcdb6'])
    norm = matplotlib.colors.BoundaryNorm([-.5, .5, 1.5, 2.5], colors.N)
    mesh = axis.pcolormesh(s, b, regimes, cmap=colors, norm=norm, shading='nearest')
    for label, validation, marker in SPLITS:
        selected = (data['status'] == 'OK') & (data['is_validation'] == validation)
        axis.scatter(data['S'][selected], data['B'][selected], marker=marker,
                     s=22, color='#202020', label=f'T4 sampled points ({label})')
    colorbar = figure.colorbar(mesh, ax=axis, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(['launch-bound', 'memory-bound', 'compute-bound'])
    axis.set(xlabel='Image side S (pixels)', ylabel='Batch size B (images)',
             yscale='log', xlim=(sizes.min() - 8, sizes.max() + 8),
             ylim=(.8, batches.max() * 1.15),
             title=f'Model regimes: launch {counts[0]}, memory {counts[1]}, compute {counts[2]} sampled points')
    axis.legend(fontsize=8, loc='upper left')
    figure.text(.5, .015, 'Color = largest of 17a, Q/W, F/C; markers = measurement locations, not measured regimes',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .035, 1, 1))
    figure.savefig(path, dpi=160)
    matplotlib.pyplot.close(figure)


def plot_flops(data, path):
    # Meta tensors propagate actual network shapes without allocating image batches
    model = CNN().float().eval().to('meta')
    counts = []

    def count(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            kernel_height, kernel_width = module.kernel_size
            counts.append(2 * output.numel() * (module.in_channels // module.groups)
                          * kernel_height * kernel_width)
        elif isinstance(module, torch.nn.Linear):
            counts.append(2 * output.numel() * module.in_features)
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            # Global average: H*W-1 additions and one division per channel
            counts.append(inputs[0].numel())

    handles = [module.register_forward_hook(count) for module in model.modules()]
    sizes = numpy.unique(data['S'])
    figure, axes = matplotlib.pyplot.subplots(3, 4, figsize=(16, 9.9))
    try:
        with torch.inference_mode():
            for axis, size in zip(axes.flat, sizes):
                selected = data['S'] == size
                batches = numpy.sort(data['B'][selected])
                counted = []
                for batch in batches:
                    counts.clear()
                    model(torch.empty((int(batch), 3, int(size), int(size)), device='meta'))
                    counted.append(sum(counts))
                predicted = flops(size, batches)
                numpy.testing.assert_array_equal(counted, predicted)
                axis.plot(batches, predicted, label='analytical prediction')
                axis.scatter(batches, counted, marker='x', color='tab:orange', label='network operation count')
                axis.set(title=f'S={int(size)} pixels', xlabel='Batch B',
                         ylabel='FLOPs per forward', xscale='log', yscale='log')
        for axis in list(axes.flat)[len(sizes):]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        save_figure(figure, path)
    finally:
        for handle in handles:
            handle.remove()


def run(args):
    results = Path(args.results)
    data = load_measurements(results / 'measurements.csv')
    if not (valid_points(data, 'latency') & data['is_validation']).any():
        raise ValueError('Need at least one valid held-out latency measurement')
    latency_theta = fit_latency(data)
    theta = {'latency': latency_theta, 'energy': fit_energy(data, latency_theta)}
    predictions = predict(data, theta)
    report = evaluate(data, predictions)

    figures = results / 'figures'
    figures.mkdir(exist_ok=True)
    for quantity, predicted in predictions.items():
        plot_parity(data, predicted, quantity, figures / f'{quantity}_parity.png')
        plot_grid(data, predicted, quantity, figures / f'{quantity}_grid.png')
    plot_oom(data, figures / 'oom_grid.png')
    plot_flops(data, figures / 'flops_grid.png')
    plot_regimes(data, theta['latency'], figures / 'regime_map.png')

    (results / 'theta.json').write_text(json.dumps(theta, indent=2, allow_nan=False))
    write_readme_section(results.parent / 'README.md', 'metrics',
                         '## Results summary\n\n```json\n'
                         + json.dumps(report, indent=2, allow_nan=False) + '\n```')
    if theta['energy'] is None:
        print('Energy fitting skipped: fewer than four valid training energy measurements')
    print(json.dumps(report, indent=2))
    print(f'Saved parameters to {results / "theta.json"} and plots to {figures}')


def parse_args():
    parser = argparse.ArgumentParser(description='Fit the CNN performance model and plot held-out predictions')
    parser.add_argument('--results', default='results')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
