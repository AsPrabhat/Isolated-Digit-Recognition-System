# audio_processing.py
import numpy as np
import speechpy
from scipy.io import wavfile
import os

def compute_mfcc(audio_path, num_cepstral=13, frame_length=0.025, 
                 frame_stride=0.01, num_filters=40, fft_length=512, 
                 low_frequency=0, high_frequency=None):
    """
    Computes MFCC features from an audio file.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None
        
    try:
        sampling_frequency, signal = wavfile.read(audio_path)

        signal = signal.astype(np.float32)
        signal = signal - np.mean(signal)

        mfcc_feat = speechpy.feature.mfcc(signal, 
                                          sampling_frequency=sampling_frequency, 
                                          frame_length=frame_length, 
                                          frame_stride=frame_stride, 
                                          num_cepstral=num_cepstral, 
                                          num_filters=num_filters, 
                                          fft_length=fft_length,
                                          low_frequency=low_frequency,
                                          high_frequency=high_frequency)

        mfcc_feat_cmvn = speechpy.processing.cmvnw(mfcc_feat, win_size=301, variance_normalization=False)
        return mfcc_feat_cmvn
    except Exception as e:
        print(f"Error processing MFCC for {audio_path}: {e}")
        return None

def load_mfcc_from_paths(audio_paths_dict, mfcc_params=None):
    """
    Loads MFCCs for a dictionary of audio file paths.
    """
    if mfcc_params is None:
        mfcc_params = {} 

    mfcc_data_dict = {}
    print(f"Processing MFCCs...")
    for digit_name, audio_path in audio_paths_dict.items():
        # print(f"  Computing MFCC for {digit_name} from {audio_path}...") # Verbose
        mfccs = compute_mfcc(audio_path, **mfcc_params)
        # if mfccs is not None: # compute_mfcc already returns None on error
        mfcc_data_dict[digit_name] = mfccs
        # else:
            # print(f"    Could not compute MFCCs for {digit_name}. File might be missing or corrupted.")
            # mfcc_data_dict[digit_name] = None # Explicitly mark as None
    print("MFCC Processing Done for this set.")
    return mfcc_data_dict


if __name__ == '__main__':
    print("Running Audio Processing Example...")
    sample_rate = 16000
    duration = 1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    dummy_signal = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    if not os.path.exists("data"):
        os.makedirs("data")
    dummy_wav_path = "data/dummy_test_audio.wav"
    wavfile.write(dummy_wav_path, sample_rate, dummy_signal)
    print(f"Created dummy WAV file: {dummy_wav_path}")

    mfcc_features = compute_mfcc(dummy_wav_path, fft_length=1103)
    if mfcc_features is not None:
        print(f"MFCC features shape for dummy audio: {mfcc_features.shape}")
    else:
        print("Failed to compute MFCC for dummy audio.")
    
    example_paths = {"dummy": dummy_wav_path}
    mfcc_dict = load_mfcc_from_paths(example_paths, mfcc_params={'fft_length': 1103})
    if "dummy" in mfcc_dict and mfcc_dict["dummy"] is not None:
        print(f"MFCC for 'dummy' via dict loading: {mfcc_dict['dummy'].shape}")

    print("Audio Processing Example Finished.")