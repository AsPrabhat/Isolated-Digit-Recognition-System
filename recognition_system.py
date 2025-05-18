# recognition_system.py
from dtw_core import DTW 

def isolated_digit_recognition(train_mfcc_dict, test_mfcc_sequence):
    """
    Recognizes a digit from a test MFCC sequence by comparing it against 
    a dictionary of training MFCC sequences using DTW.
    """
    if not train_mfcc_dict or test_mfcc_sequence is None:
        # print("Error: Training data or test sequence is empty/None for recognition.") # Can be too verbose
        return None, float('inf')
        
    min_dtw_distance = float('inf')
    recognized_digit_name = None
    
    for digit_name, reference_mfcc_sequence in train_mfcc_dict.items():
        if reference_mfcc_sequence is None:
            # print(f"Warning: Reference MFCC for training digit '{digit_name}' is None. Skipping.") # Can be too verbose
            continue
            
        distance, _, _ = DTW(test_mfcc_sequence, reference_mfcc_sequence)
        
        if distance < min_dtw_distance:
            min_dtw_distance = distance
            recognized_digit_name = digit_name
            
    return recognized_digit_name, min_dtw_distance

if __name__ == '__main__':
    import numpy as np
    print("Running Recognition System Example...")

    dummy_train_mfcc = {
        "one": np.random.rand(50, 13), 
        "two": np.random.rand(55, 13)
    }
    dummy_test_mfcc_one = np.random.rand(52, 13)

    recognized, dist = isolated_digit_recognition(dummy_train_mfcc, dummy_test_mfcc_one)
    if recognized:
        print(f"Test sample (likely 'one') recognized as: '{recognized}' with distance: {dist:.2f}")
    else:
        print("Recognition failed for example.")
    
    print("Recognition System Example Finished.")