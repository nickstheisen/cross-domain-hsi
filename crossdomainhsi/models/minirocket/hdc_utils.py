import numpy as np
from numpy.fft import fft, ifft, ifftshift


def fpe_orig(inputs, phases, beta):
    """
    Fractional binding for scalar encoding using the original method.

    Args:
        inputs: Scalar input vector.
        phases: Seed vector [#dim].
        beta: Scaling factor for similarity.

    Returns:
        HDC vector for each scalar value.
    """
    inputs = np.array(inputs, dtype=np.float32)

    # Prepare phase vector
    D = len(phases)
    phases = phases[:((D - 1) // 2)]  # First half of the phase values
    if D % 2 == 1:
        phases = np.concatenate((phases, [0], -phases[::-1]))
    else:
        phases = np.concatenate(([0], phases, [0], -phases[::-1]))

    # Generate base vector using inverse FFT
    base_fft = np.exp(1j * phases)
    base = np.real(ifft(ifftshift(base_fft)))

    # Compute fractional power expansion (FPE)
    exponent = beta * inputs[:, None] + 1
    fpe = np.real(ifft(np.power(fft(base[None, :]), exponent)))

    # Standardize the results
    fpe = (fpe.T - np.mean(fpe, axis=1)).T / np.std(fpe, axis=1, keepdims=True)
    return fpe.astype(np.float32)


def fpe_hrr(inputs, phases, beta):
    """
    Fractional binding for scalar encoding using HRR.

    Args:
        inputs: Scalar input vector.
        phases: Seed vector [#dim].
        beta: Scaling factor for similarity.

    Returns:
        HDC vector for each scalar value.
    """
    inputs = np.array(inputs, dtype=np.float32)
    return _create_base_fpe(inputs, phases, beta)


def fpe_fhrr(inputs, phases, beta):
    """
    Fractional binding for scalar encoding using FHRR.

    Args:
        inputs: Scalar input vector.
        phases: Seed vector [#dim].
        beta: Scaling factor for similarity.

    Returns:
        Array of computed angles.
    """
    inputs = np.array(inputs, dtype=np.float32)
    return inputs[:, None] * phases * beta


def fpe_sinusoid(inputs, phases, bandwidth):
    """
    Sinusoidal fractional binding for scalar encoding.

    Args:
        inputs: Scalar input vector.
        phases: Seed vector [#dim].
        bandwidth: Scaling factor for similarity.

    Returns:
        Fourier feature encoding using sine and cosine.
    """
    inputs = np.array(inputs, dtype=np.float32)
    exponent = inputs * bandwidth + 1
    output = phases * exponent[:, None]

    mid_idx = phases.shape[0] // 2
    sin_part = np.sin(output)[:, :mid_idx]
    cos_part = np.cos(output)[:, :mid_idx]
    return np.concatenate((cos_part, sin_part), axis=1)


def _create_base_fpe(inputs, phases, beta):
    """
    Helper function for computing base FPE with HRR and similar methods.
    """
    inputs = np.array(inputs, dtype=np.float32)

    # Prepare phase vector
    D = len(phases)
    phases = phases[:((D - 1) // 2)]
    if D % 2 == 1:
        phases = np.concatenate((phases, [0], -phases[::-1]))
    else:
        phases = np.concatenate(([0], phases, [0], -phases[::-1]))

    # Generate base vector and compute FPE
    base_fft = np.exp(1j * phases)
    base = np.real(ifft(ifftshift(base_fft)))
    exponent = beta * inputs[:, None]
    return np.real(ifft(np.power(fft(base[None, :]), exponent)))


def create_pose_matrix(num_poses, scale, HDC_dim, seed=0, fpe_method='orig', kernel='sinc'):
    """
    Encode poses (e.g., timestamps or positions) into a pose matrix.

    Args:
        num_poses: Number of scalar values (e.g., poses).
        scale: Scale factor for fractional binding (similarity decrease).
        HDC_dim: Dimensionality of the HDC space.
        seed: Random seed for generating initial phases.
        fpe_method: Method for fractional binding ('orig', 'sinusoid', 'cosine', etc.).
        kernel: Kernel type for initialization ('sinc', 'gaussian', 'triangular').

    Returns:
        Pose matrix encoding the input scalars.
    """
    np.random.seed(seed)

    # Initialize phase vector based on kernel type
    if kernel == 'sinc':
        init_vector = np.random.uniform(-np.pi, np.pi, HDC_dim)
    elif kernel == 'gaussian':
        init_vector = np.random.normal(0, 1, HDC_dim)
        scale *= 3
    elif kernel == 'triangular':
        p = np.power(np.sinc(np.linspace(-np.pi, np.pi, HDC_dim)), 2)
        init_vector = np.random.choice(np.linspace(-np.pi, np.pi, HDC_dim), HDC_dim, p=p / p.sum())
        scale *= 6
    else:
        raise ValueError('Invalid kernel type')

    # Generate input time space
    time_space = np.linspace(0, 1, num_poses)

    # Apply the specified FPE method
    fpe_methods = {
        'orig': lambda: fpe_orig(time_space, init_vector, scale),
        'sinusoid': lambda: fpe_sinusoid(time_space, init_vector, scale),
        'cosine': lambda: fpe_cosine(time_space, init_vector, scale, np.random.uniform(-np.pi, np.pi, HDC_dim)),
    }

    if fpe_method in fpe_methods:
        poses = fpe_methods[fpe_method]()
    else:
        raise ValueError('Invalid FPE method.')

    return poses.astype(np.float32)