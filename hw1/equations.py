import numpy

# Each FP32 value occupies 4 bytes
DTYPE_BYTES = 4

# One fused multiply add counts as one multiplication and one addition
FLOPS_PER_FMA = 2

CONVS = [
    # (in_channels, out_channels, kernel_size, padding, stride)
    (3, 32, 7, 3, 2),
    (32, 64, 5, 2, 1),
    (64, 128, 3, 1, 2),
    (128, 256, 1, 0, 1),
    (256, 256, 3, 1, 2),
    (256, 512, 1, 0, 1),
]

# Each convolution has ci * co * k² weights
# The two linear layers have 512 * 256 and 256 * 100 weights
WEIGHTS = sum(ci*co*k*k for ci,co,k,_,_ in CONVS) + 512*256 + 256*100
PARAMETER_BYTES = DTYPE_BYTES * WEIGHTS


def _inputs(image_size, batch):
    # Broadcast image sizes S and batch sizes B to a common shape for grid calculations
    s, b = numpy.broadcast_arrays(numpy.asarray(image_size, dtype=float), numpy.asarray(batch, dtype=float))
    if numpy.any((s < 16) | (s % 16 != 0) | (b < 1) | (b % 1 != 0)):
        raise ValueError('S must be a positive multiple of 16; B a positive integer')
    return s, b


def flops(image_size, batch):
    s, b = _inputs(image_size, batch)
    total = numpy.zeros_like(s)
    size = s
    # ReLU and MaxPool comparisons are excluded from the FLOP count
    for i, (in_channels, out_channels, kernel_size, padding, stride) in enumerate(CONVS):
        # Output side length: floor((input side + 2 * padding - kernel) / stride) + 1
        size = numpy.floor((size + 2 * padding - kernel_size) / stride) + 1
        output_elements = b * out_channels * size**2
        # Each output uses in_channels * kernel_size² multiply-accumulates, at 2 FLOPs each
        total += FLOPS_PER_FMA * output_elements * in_channels * kernel_size**2
        if i == 0:
            # MaxPool after Conv1: kernel 3, padding 1, stride 2
            size = numpy.floor((size + 2 * 1 - 3) / 2) + 1
    # GlobalAvgPool: size² - 1 additions + 1 division for each of 512 channels per image
    total += b * 512 * size**2
    # Linear 512 -> 256: 512 multiply-accumulates for each of 256 outputs per image
    total += FLOPS_PER_FMA * b * 512 * 256
    # Linear 256 -> 100: 256 multiply-accumulates for each of 100 outputs per image
    total += FLOPS_PER_FMA * b * 256 * 100
    return total


def bytes_moved(image_size, batch):
    s, b = _inputs(image_size, batch)
    # Input contains B RGB images with S * S pixels and 3 channels
    input_elements = 3 * b * s**2
    total = numpy.zeros_like(s)
    size = s
    for i, (in_channels, out_channels, kernel_size, padding, stride) in enumerate(CONVS):
        size = numpy.floor((size + 2 * padding - kernel_size) / stride) + 1
        output_elements = b * out_channels * size**2
        weights = in_channels * out_channels * kernel_size**2
        # Ideal convolution traffic: read input and weights once, write output once
        total += DTYPE_BYTES * (input_elements + weights + output_elements)
        # In-place ReLU reads and overwrites each output value: 2 transfers per value
        total += 2 * DTYPE_BYTES * output_elements
        input_elements = output_elements
        if i == 0:
            # MaxPool 3x3, padding 1, stride 2: read its input and write its output
            size = numpy.floor((size + 2 * 1 - 3) / 2) + 1
            output_elements = b * out_channels * size**2
            total += DTYPE_BYTES * (input_elements + output_elements)
            input_elements = output_elements
    # GlobalAvgPool reads the final feature maps and writes 512 values per image
    total += DTYPE_BYTES * (input_elements + b * 512)
    # Linear 512 -> 256: B*512 input values, 512*256 weights, B*256 output values
    total += DTYPE_BYTES * (b * 512 + 512 * 256 + b * 256)
    # Head ReLU reads and writes B*256 values
    total += 2 * DTYPE_BYTES * b * 256
    # Linear 256 -> 100: B*256 input values, 256*100 weights, B*100 output values
    total += DTYPE_BYTES * (b * 256 + 256 * 100 + b * 100)
    return total


def memory(image_size, batch):
    # Retain the input and all activation tensors
    s, b = _inputs(image_size, batch)

    # Input contains B images with 3 channels and S * S pixels
    total_elements = b * 3 * s**2
    size = s

    for i, (in_channels, out_channels, kernel_size, padding, stride) in enumerate(CONVS):
        # Retain each convolution output
        size = numpy.floor((size + 2 * padding - kernel_size) / stride) + 1
        total_elements += b * out_channels * size**2

        if i == 0:
            # Retain the MaxPool output after Conv1: kernel 3, padding 1, stride 2
            size = numpy.floor((size + 2 * 1 - 3) / 2) + 1
            total_elements += b * out_channels * size**2

    # GlobalAvgPool produces 512 values per image
    total_elements += b * 512
    # Linear 512 -> 256 produces 256 values per image
    total_elements += b * 256
    # Linear 256 -> 100 produces 100 values per image
    total_elements += b * 100

    # In-place ReLU and flatten share storage and are not counted again
    return PARAMETER_BYTES + DTYPE_BYTES * total_elements


def latency(image_size, batch, theta):
    # Arithmetic operations divided by effective FLOP/s gives compute time in seconds
    compute_time = flops(image_size, batch) / theta['compute_flops_per_second']

    # Bytes transferred divided by effective bytes/s gives transfer time in seconds
    transfer_time = bytes_moved(image_size, batch) / theta['bandwidth_bytes_per_second']

    # 6 convolutions + 7 ReLUs + MaxPool + GlobalAvgPool + 2 linear layers
    num_operations = 6 + 7 + 1 + 1 + 2
    overhead_time = num_operations * theta['launch_seconds']

    # Whole-network approximation: overhead plus the larger compute or transfer cost
    return overhead_time + numpy.maximum(compute_time, transfer_time)


def energy(image_size, batch, theta_energy):
    f = flops(image_size, batch)
    q = bytes_moved(image_size, batch)
    t = latency(image_size, batch, theta_energy['latency'])
    # Joules = time-proportional power * seconds + J/FLOP * FLOPs + J/byte * bytes
    return theta_energy['watts']*t + theta_energy['joules_per_flop']*f + theta_energy['joules_per_byte']*q
