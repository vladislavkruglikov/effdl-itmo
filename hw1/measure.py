import argparse
import csv
import gc
import json
import platform
import random
import statistics
import threading
import time
from pathlib import Path

import numpy
import torch

from equations import memory
from models import CNN


BASE_IMAGE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CSV_FIELDS = [
    'S', 'B', 'is_validation', 'status', 'latency', 'memory', 'energy',
    'predicted_memory', 'free_bytes_before', 'energy_repeats', 'power_samples',
    'energy_loop_latency', 'energy_method', 'error',
]


def write_readme_section(path, name, content):
    path = Path(path)
    text = path.read_text() if path.exists() else '# Homework 1\n'
    start, end = f'<!-- {name}:start -->', f'<!-- {name}:end -->'
    section = f'{start}\n{content}\n{end}'
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        text = before + section + after
    else:
        text = text.rstrip() + '\n\n' + section + '\n'
    path.write_text(text)


def grid(seed=2026):
    rng = random.Random(seed)
    extra_sizes = rng.sample([s for s in range(32, 513, 16) if s not in BASE_IMAGE_SIZES], 4)
    extra_batches = rng.sample([b for b in range(1, 257) if b not in BASE_BATCH_SIZES], 3)
    sizes = sorted(BASE_IMAGE_SIZES + extra_sizes)
    batches = sorted(BASE_BATCH_SIZES + extra_batches)
    return [
        (size, batch, size not in BASE_IMAGE_SIZES or batch not in BASE_BATCH_SIZES)
        for size in sizes for batch in batches
    ]


def configure_gpu(device, seed):
    if not torch.cuda.is_available():
        raise RuntimeError('An NVIDIA CUDA GPU is required; no measurements were created')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def warmup(model, inputs, repeats):
    for _ in range(repeats):
        model(inputs)
    torch.cuda.synchronize()


@torch.inference_mode()
def measure_memory(model, inputs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    output = model(inputs)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    del output
    return peak_bytes


@torch.inference_mode()
def measure_latency(model, inputs, repeats):
    times = []
    for _ in range(repeats):
        # Synchronization includes GPU completion in the host wall-clock measurement
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(inputs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        del output
    return statistics.median(times)


@torch.inference_mode()
def run_energy_window(model, inputs, seconds):
    torch.cuda.synchronize()
    repeats = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        model(inputs)
        torch.cuda.synchronize()
        repeats += 1
    end = time.perf_counter()
    if repeats == 0:
        raise RuntimeError('No forward passes completed in the energy window')
    return start, end, repeats


class EnergyMeter:
    def __init__(self, device, index=None):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        try:
            # CUDA device order may differ from physical NVML indices
            uuid = getattr(torch.cuda.get_device_properties(device), 'uuid', None)
            if index is not None:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                if uuid and str(uuid) != pynvml.nvmlDeviceGetUUID(self.handle):
                    raise RuntimeError('NVML index does not match the CUDA device UUID')
            elif uuid:
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            else:
                raise RuntimeError('CUDA UUID unavailable; supply the physical --nvml-index')
            try:
                self.read_energy()
                self.counter_supported = True
            except pynvml.NVMLError:
                self.counter_supported = False
                self.read_power()
        except Exception:
            self.close()
            raise

    def read_energy(self):
        # NVML reports cumulative whole-GPU energy in millijoules
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000

    def read_power(self):
        # NVML reports whole-GPU power in milliwatts
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000

    def close(self):
        self.nvml.nvmlShutdown()

    def measure(self, model, inputs, seconds):
        if self.counter_supported:
            torch.cuda.synchronize()
            before = self.read_energy()
            start, end, repeats = run_energy_window(model, inputs, seconds)
            joules = self.read_energy() - before
            if joules > 0:
                return self.result(joules, start, end, repeats, 'nvml_counter', 0)
        # An unsupported or unchanged counter requires a new power-sampled window
        return self.measure_power(model, inputs, seconds)

    def measure_power(self, model, inputs, seconds):
        samples = [(time.perf_counter(), self.read_power())]
        errors = []
        stop = threading.Event()

        def sample_power():
            try:
                while not stop.wait(0.02):
                    samples.append((time.perf_counter(), self.read_power()))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=sample_power, daemon=True)
        thread.start()
        try:
            start, end, repeats = run_energy_window(model, inputs, seconds)
        finally:
            # Always stop sampling, including when a forward raises OOM
            stop.set()
            thread.join()
        if errors:
            raise RuntimeError('NVML power sampling failed') from errors[0]
        samples.append((time.perf_counter(), self.read_power()))
        times, powers = numpy.asarray(samples).T
        inside = (times > start) & (times < end)
        window_times = numpy.r_[start, times[inside], end]
        window_powers = numpy.interp(window_times, times, powers)
        # Integrate watts over seconds using trapezoids, without subtracting idle power
        joules = numpy.sum(numpy.diff(window_times) * (window_powers[:-1] + window_powers[1:]) / 2)
        if not numpy.isfinite(joules) or joules <= 0:
            raise RuntimeError('NVML did not provide a positive finite energy measurement')
        return self.result(joules, start, end, repeats, 'nvml_power', len(samples))

    @staticmethod
    def result(joules, start, end, repeats, method, samples):
        return {
            'energy': float(joules / repeats),
            'energy_repeats': repeats,
            'energy_loop_latency': (end - start) / repeats,
            'energy_method': method,
            'power_samples': samples,
        }


def environment_info(args, meter):
    properties = torch.cuda.get_device_properties(args.device)
    info = {
        'gpu': properties.name,
        'total_memory': properties.total_memory,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'args': vars(args),
        'flags': {'benchmark': False, 'cudnn_tf32': False, 'matmul_tf32': False},
        'latency_method': 'median synchronized host wall-clock time',
        'energy_method': 'NVML total-energy counter, falling back to integrated power',
    }
    if meter is not None:
        info['driver'] = meter.nvml.nvmlSystemGetDriverVersion()
        info['gpu_uuid'] = meter.nvml.nvmlDeviceGetUUID(meter.handle)
        info['energy_counter_supported'] = meter.counter_supported
    return info


def measure_configuration(image_size, batch, is_validation, args, meter):
    row = {
        'S': image_size,
        'B': batch,
        'is_validation': int(is_validation),
        'predicted_memory': float(memory(image_size, batch)),
    }
    model = inputs = None
    try:
        clear_memory()
        row['free_bytes_before'] = torch.cuda.mem_get_info()[0]
        model = CNN().cuda().float().eval()
        with torch.inference_mode():
            inputs = torch.randn(batch, 3, image_size, image_size, device='cuda', dtype=torch.float32)
        warmup(model, inputs, args.warmup)
        row['memory'] = measure_memory(model, inputs)
        row['latency'] = measure_latency(model, inputs, args.repeats)
        if meter is not None:
            row.update(meter.measure(model, inputs, args.energy_seconds))
        row['status'] = 'OK'
    except torch.cuda.OutOfMemoryError:
        row['status'] = 'OOM'
    except Exception as error:
        row.update(status='ERROR', error=f'{type(error).__name__}: {error}')
    finally:
        model = inputs = None
        clear_memory()
    return row


def run(args):
    output = Path(args.output)
    destination = output / 'measurements.csv'
    if destination.exists():
        raise FileExistsError('Measurements already exist; choose another --output directory')
    configure_gpu(args.device, args.seed)
    meter = None
    try:
        if not args.skip_energy:
            meter = EnergyMeter(args.device, args.nvml_index)
        output.mkdir(parents=True, exist_ok=True)
        metadata = environment_info(args, meter)
        write_readme_section(output.parent / 'README.md', 'environment',
                             '## Measured GPU / software environment\n\n```json\n'
                             + json.dumps(metadata, indent=2) + '\n```')
        points = grid(args.seed)
        random.Random(args.seed + 1).shuffle(points)
        print(f'GPU: {metadata["gpu"]} | Configurations: {len(points)}', flush=True)
        # Exclusive creation preserves existing results even if another process starts a run
        with destination.open('x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            file.flush()
            for index, (size, batch, validation) in enumerate(points, start=1):
                row = measure_configuration(size, batch, validation, args, meter)
                writer.writerow(row)
                file.flush()
                print(f'[{index}/{len(points)}] S={size} B={batch}: {row["status"]}', flush=True)
    finally:
        if meter is not None:
            meter.close()
    print(f'Saved {destination}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Measure CNN latency, memory and energy on an NVIDIA GPU')
    parser.add_argument('--output', default='results')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--nvml-index', type=int)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--energy-seconds', type=float, default=3)
    parser.add_argument('--skip-energy', action='store_true', help='Partial run without energy measurements')
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or not numpy.isfinite(args.energy_seconds) or args.energy_seconds < 2:
        parser.error('Require warmup/repeats >= 1 and finite energy-seconds >= 2')
    return args


if __name__ == '__main__':
    run(parse_args())
