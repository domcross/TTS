# from https://github.com/librosa/librosa/blob/main/librosa/core/fft.py
# from https://github.com/librosa/librosa/blob/main/librosa/util/utils.py

import importlib
use_cupy = importlib.util.find_spec("cupy") is not None
if use_cupy:
    print("{} using cupy".format(__name__))
    import cupy as np
else:
    import numpy as np
import scipy

MAX_MEM_BLOCK = 2**8 * 2**10


def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):
    #print("enter stft...")
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = __get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = __pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    __valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = __frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                        dtype=dtype,
                        order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] *
                                    stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        stft_matrix[:, bl_s:bl_t] = np.fft.rfft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)
    #print("...leaving stft")
    return stft_matrix


def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
            center=True, dtype=np.float32, length=None):
    #print("istft enter...")
    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = __get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add a broadcasting axis
    ifft_window = __pad_center(ifft_window, n_fft)[:, np.newaxis]

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(
            stft_matrix.shape[1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[1]

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    
    #MAX_MEM_BLOCK = 2**8 * 2**10
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] *
                                    stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    #fft = get_fftlib()

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * np.fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[frame * hop_length:], ytmp, hop_length)

        frame += (bl_t - bl_s)

    # Normalize by sum of squared window
    ifft_window_sum = __window_sumsquare(window,
                                    n_frames,
                                    win_length=win_length,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    dtype=dtype)

    approx_nonzero_indices = ifft_window_sum > __tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2):-int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = __fix_length(y[start:], length)

    return y


def __get_window(window, win_length, fftbins=True):
    if use_cupy:
        return np.asarray(scipy.signal.get_window(window, win_length, fftbins=fftbins))
    return scipy.signal.get_window(window, win_length, fftbins=fftbins)


def __pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                            'at least input size ({:d})').format(size, n))
    return np.pad(data, lengths, **kwargs)


def __frame(x, frame_length, hop_length, axis=-1):
    if not isinstance(x, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                            'given type(x)={}'.format(type(x)))

    if x.shape[axis] < frame_length:
        raise ParameterError('Input is too short (n={:d})'
                            ' for frame_length={:d}'.format(x.shape[axis], frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if axis == -1 and not x.flags['F_CONTIGUOUS']:
        warnings.warn('lr_custom.frame called with axis={} '
                    'on a non-contiguous input. This will result in a copy.'.format(axis))
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags['C_CONTIGUOUS']:
        warnings.warn('lr_custom.frame called with axis={} '
                    'on a non-contiguous input. This will result in a copy.'.format(axis))
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)
    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = [int(strides[0])]
        strides.append(int(hop_length * new_stride))
        # = list(strides) + [hop_length * new_stride]
    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise ParameterError('Frame axis={} must be either 0 or -1'.format(axis))
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def __valid_audio(y, mono=True):
    if not isinstance(y, np.ndarray):
        raise ParameterError('Audio data must be of type numpy.ndarray')

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError('Audio data must be floating-point')

    if mono and y.ndim != 1:
        raise ParameterError('Invalid shape for monophonic audio: '
                            'ndim={:d}, shape={}'.format(y.ndim, y.shape))

    elif y.ndim > 2 or y.ndim == 0:
        raise ParameterError('Audio data must have shape (samples,) or (channels, samples). '
                            'Received shape={}'.format(y.shape))

    elif y.ndim == 2 and y.shape[0] < 2:
        raise ParameterError('Mono data must have shape (samples,). '
                            'Received shape={}'.format(y.shape))

    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')

    return True


def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[0]
    for frame in range(ytmp.shape[1]):
        sample = frame * hop_length
        y[sample:(sample + n_fft)] += ytmp[:, frame]

def __window_sumsquare(window, n_frames, hop_length=512, win_length=None, n_fft=2048,
                     dtype=np.float32, norm=None):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = __get_window(window, win_length)
    win_sq = __normalize(win_sq, norm=norm)**2
    win_sq = __pad_center(win_sq, n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x

def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    '''Helper function for window sum-square calculation.'''
    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]

def __fix_length(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data

def __normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    # Avoid div-by-zero
    if threshold is None:
        threshold = __tiny(S)

    elif threshold <= 0:
        raise ParameterError('threshold={} must be strictly '
                             'positive'.format(threshold))

    if fill not in [None, False, True]:
        raise ParameterError('fill={} must be None or boolean'.format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError('Input must be finite')

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError('Cannot normalize with norm=0 and fill=True')

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)

        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)

    elif norm is None:
        return S

    else:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def __tiny(x):
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny