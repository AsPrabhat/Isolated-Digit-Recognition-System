# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from audio_processing import compute_mfcc
from recognition_system import isolated_digit_recognition

def calculate_accuracy_and_confusion_matrix(train_mfcc_dict, 
                                            test_sets_mfcc_list, 
                                            digits_ordered):
    """
    Calculates accuracy and confusion matrix for digit recognition.
    """
    num_digits = len(digits_ordered)
    total_correct_recognitions = 0
    total_test_samples = 0
    confusion_mat_array = np.zeros((num_digits, num_digits), dtype=int)
    digit_to_index = {name: i for i, name in enumerate(digits_ordered)}

    if not train_mfcc_dict or all(v is None for v in train_mfcc_dict.values()):
        print("Error: Training MFCC data is empty or all None for evaluation.")
        return 0.0, confusion_mat_array
    
    for test_set_index, current_test_mfcc_set in enumerate(test_sets_mfcc_list):
        # print(f"Evaluating Test Set {test_set_index + 1}...") # Verbose
        if not current_test_mfcc_set:
            # print(f"  Warning: Test Set {test_set_index + 1} is empty. Skipping.") # Verbose
            continue
            
        for true_digit_name in digits_ordered:
            test_mfcc = current_test_mfcc_set.get(true_digit_name) # Use .get()
            if test_mfcc is None:
                continue 

            total_test_samples += 1
            
            recognized_digit_name, _ = isolated_digit_recognition(train_mfcc_dict, test_mfcc)

            if recognized_digit_name is None:
                # print(f"  Recognition failed for test sample of '{true_digit_name}'.") # Verbose
                continue

            true_label_idx = digit_to_index[true_digit_name]
            predicted_label_idx = digit_to_index.get(recognized_digit_name, -1)

            if predicted_label_idx != -1:
                confusion_mat_array[true_label_idx, predicted_label_idx] += 1
            # else:
                # print(f"  Warning: Reco digit '{recognized_digit_name}' not in map.") # Verbose

            if recognized_digit_name == true_digit_name:
                total_correct_recognitions += 1
    
    accuracy = (total_correct_recognitions / total_test_samples) if total_test_samples > 0 else 0.0
    return accuracy, confusion_mat_array


def plot_confusion_matrix(confusion_mat_array, digits_ordered, accuracy, save_path=None):
    """ Plots and optionally saves the confusion matrix. """
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat_array, annot=True, fmt="d", cmap="Blues", 
                xticklabels=digits_ordered, yticklabels=digits_ordered)
    plt.title(f'Confusion Matrix for Digit Recognition\nAccuracy: {accuracy*100:.2f}%')
    plt.ylabel('True Digit')
    plt.xlabel('Predicted Digit')
    plt.tight_layout()

    if save_path:
        plots_dir = os.path.dirname(save_path)
        if plots_dir and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Confusion matrix plot saved to {save_path}")
        return save_path
    else:
        plt.show()
        return None


def load_tdigits_reference_mfcc(base_path, speaker_id, digits_list, rep_id, mfcc_params=None):
    reference_mfcc_dict = {}
    print(f"Loading TDIGITS reference MFCCs for speaker '{speaker_id}', repetition '{rep_id}'...")
    
    if mfcc_params is None:
        mfcc_params = {}

    for digit in digits_list:
        audio_filename = f"{digit}_{speaker_id}_{rep_id}.wav"
        audio_path = os.path.join(base_path, speaker_id, audio_filename)
        
        # print(f"  Attempting to load: {audio_path}") # For debugging
        if not os.path.exists(audio_path):
            print(f"    --> File NOT FOUND: {audio_path}")
            reference_mfcc_dict[digit] = None
            continue
        # else: # For debugging
            # print(f"    --> File FOUND: {audio_path}")
            
        mfccs = compute_mfcc(audio_path, **mfcc_params)
        # if mfccs is None: # For debugging
            # print(f"    --> MFCC computation FAILED for: {audio_path}")
        reference_mfcc_dict[digit] = mfccs
    print("TDIGITS Reference MFCCs loaded for this speaker.")
    return reference_mfcc_dict

def evaluate_on_tdigits(reference_mfccs, test_speaker_id, digits_list, base_path, 
                        num_test_repetitions=50, reference_repetition_id='0', 
                        reference_speaker_id="ref_speaker", mfcc_params=None):
    total_correct = 0
    total_tested = 0
    
    if mfcc_params is None:
        mfcc_params = {}

    print(f"Evaluating TDIGITS: Ref '{reference_speaker_id}', Test '{test_speaker_id}'")
    if not reference_mfccs or all(v is None for v in reference_mfccs.values()):
        print(f"Error: Ref MFCCs missing/empty for speaker '{reference_speaker_id}'. Cannot evaluate {test_speaker_id}.")
        return 0.0, 0, 0

    for true_digit in digits_list:
        for rep_idx in range(num_test_repetitions):
            if test_speaker_id == reference_speaker_id and str(rep_idx) == reference_repetition_id:
                continue
            
            test_audio_filename = f"{true_digit}_{test_speaker_id}_{rep_idx}.wav"
            test_audio_path = os.path.join(base_path, test_speaker_id, test_audio_filename)

            if not os.path.exists(test_audio_path):
                continue

            test_mfcc = compute_mfcc(test_audio_path, **mfcc_params)
            if test_mfcc is None:
                continue
            
            total_tested += 1
            recognized_digit, _ = isolated_digit_recognition(reference_mfccs, test_mfcc)

            if recognized_digit == true_digit:
                total_correct += 1

    accuracy = (total_correct / total_tested) if total_tested > 0 else 0.0
    print(f"  Results: {total_correct}/{total_tested}. Acc: {accuracy*100:.2f}%")
    return accuracy, total_correct, total_tested

if __name__ == '__main__':
    print("Running Evaluation Module Example (Illustrative)...")
    dummy_train_mfcc = {"0": np.random.rand(50,13), "1": np.random.rand(50,13)}
    dummy_test_set1 = {"0": np.random.rand(50,13), "1": np.random.rand(50,13)} 
    dummy_digits = ["0", "1"]

    acc, conf_mat_arr = calculate_accuracy_and_confusion_matrix(
        dummy_train_mfcc, 
        [dummy_test_set1], 
        dummy_digits
    )
    print(f"Dummy Accuracy: {acc*100:.2f}%")
    print(f"Dummy Confusion Matrix:\n{conf_mat_arr}")
    
    if not os.path.exists("plots"):
        os.makedirs("plots")
    cm_save_path = "plots/evaluation_example_cm_plot.png"
    plot_confusion_matrix(conf_mat_arr, dummy_digits, acc, save_path=cm_save_path)
    print("Evaluation Module Example Finished.")