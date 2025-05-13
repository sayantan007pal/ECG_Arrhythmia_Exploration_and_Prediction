import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis


def extract_rr_intervals(ecg_signal, fs):
    """
    Extract RR intervals from ECG signal using R-peak detection.
    Returns array of RR intervals in seconds.
    """
    peaks, _ = find_peaks(ecg_signal, distance=fs*0.2)
    rr_intervals = np.diff(peaks) / fs
    return rr_intervals


def extract_features(ecg_signal, fs):
    """
    Extracts a rich set of features from an ECG segment.
    Features include RR intervals, energy, skewness, kurtosis, peak-to-peak, mean/median absolute deviation.
    """
    rr_intervals = extract_rr_intervals(ecg_signal, fs)
    # RR interval features
    rr_feats = [
        np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
        np.std(rr_intervals) if len(rr_intervals) > 0 else 0,
        np.min(rr_intervals) if len(rr_intervals) > 0 else 0,
        np.max(rr_intervals) if len(rr_intervals) > 0 else 0,
        len(rr_intervals)
    ]
    # Signal features
    energy = np.sum(ecg_signal ** 2)
    skewness = skew(ecg_signal)
    kurt = kurtosis(ecg_signal)
    ptp = np.ptp(ecg_signal)
    mad = np.mean(np.abs(ecg_signal - np.mean(ecg_signal)))
    medad = np.median(np.abs(ecg_signal - np.median(ecg_signal)))
    signal_feats = [energy, skewness, kurt, ptp, mad, medad]
    return np.array(rr_feats + signal_feats) 