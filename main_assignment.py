# main_assignment.py
import os
import numpy as np

from dtw_core import DTW, plotDTWpath
from audio_processing import load_mfcc_from_paths
from recognition_system import isolated_digit_recognition
from evaluation import (
    calculate_accuracy_and_confusion_matrix, 
    plot_confusion_matrix,
    load_tdigits_reference_mfcc,
    evaluate_on_tdigits
)
import config
from html_reporter import generate_html_report

REPORT_DATA = {
    "part_a_results": None,
    "part_b_results": None,
    "part_c_results": None,
}

def ensure_data_dirs_exist():
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if not os.path.exists(config.DATA_BASE_PATH):
        os.makedirs(config.DATA_BASE_PATH)
        print(f"Created directory: {config.DATA_BASE_PATH}")
        print(f"Please place your recorded audio files (e.g., zero_1.wav) in this directory.")
    if not os.path.exists(config.TDIGITS_BASE_PATH):
         print(f"Warning: TDIGITS directory not found: {config.TDIGITS_BASE_PATH}")
         print(f"Please download and place the TDIGITS_subset folder here.")

def part_a():
    print("\n--- Running Part (a): Basic DTW ---")
    results = {}
    sequenceA = [4, 3, 7, 0, 2, 6, 5]
    sequenceB = [3, 7, 1, 6, 1, 5, 4, 4]
    results['sequence_a'] = sequenceA
    results['sequence_b'] = sequenceB
    
    try:
        dist, path, _ = DTW(sequenceA, sequenceB)
        results['dtw_distance'] = dist
        print(f"Optimal DTW Distance between A and B: {dist:.2f}")
        
        plot_save_path = os.path.join("plots", "part_a_dtw_plot.png")
        plotDTWpath(sequenceA, sequenceB, title_prefix="Part (a) Example: ", save_path=plot_save_path)
        results['plot_path'] = plot_save_path if os.path.exists(plot_save_path) else None
    except Exception as e:
        print(f"Error in Part A: {e}")
        results['error'] = str(e)
    
    REPORT_DATA["part_a_results"] = results
    print("--- Part (a) Finished ---")

def part_b():
    print("\n--- Running Part (b): Isolated Digit Recognition (Your Recordings) ---")
    results = {}
    
    audio_paths_train = {name: os.path.join(config.DATA_BASE_PATH, f"{name}_1.wav") for name in config.DIGITS_ORDERED}
    audio_paths_test1 = {name: os.path.join(config.DATA_BASE_PATH, f"{name}_2.wav") for name in config.DIGITS_ORDERED}
    audio_paths_test2 = {name: os.path.join(config.DATA_BASE_PATH, f"{name}_3.wav") for name in config.DIGITS_ORDERED}
    audio_paths_test3 = {name: os.path.join(config.DATA_BASE_PATH, f"{name}_4.wav") for name in config.DIGITS_ORDERED}

    sample_train_file = audio_paths_train.get(config.DIGITS_ORDERED[0])
    if not sample_train_file or not os.path.exists(sample_train_file):
        error_msg = f"Sample training file {sample_train_file or 'N/A'} not found."
        print(f"ERROR: {error_msg} Skipping Part (b).")
        results["error"] = error_msg
        REPORT_DATA["part_b_results"] = results
        return

    print("Extracting MFCCs for your recordings...")
    train_mfcc = load_mfcc_from_paths(audio_paths_train, config.MFCC_PARAMS)
    test1_mfcc = load_mfcc_from_paths(audio_paths_test1, config.MFCC_PARAMS)
    test2_mfcc = load_mfcc_from_paths(audio_paths_test2, config.MFCC_PARAMS)
    test3_mfcc = load_mfcc_from_paths(audio_paths_test3, config.MFCC_PARAMS)

    if not train_mfcc or all(v is None for v in train_mfcc.values()):
        error_msg = "MFCC extraction failed for training data."
        print(f"ERROR: {error_msg} Skipping further Part (b).")
        results["error"] = error_msg
        REPORT_DATA["part_b_results"] = results
        return
    results['train_mfcc_loaded'] = True


    example_rec_data = {}
    test_digit_label_ex = config.DIGITS_ORDERED[0] 
    if test_digit_label_ex in test1_mfcc and test1_mfcc.get(test_digit_label_ex) is not None:
        test_sample_mfcc_ex = test1_mfcc[test_digit_label_ex]
        rec_digit_ex, dist_ex = isolated_digit_recognition(train_mfcc, test_sample_mfcc_ex)
        example_rec_data = {
            "test_label": test_digit_label_ex,
            "recognized_digit": rec_digit_ex,
            "distance": dist_ex if rec_digit_ex is not None else float('inf')
        }
        print(f"\nExample Recognition: Test '{test_digit_label_ex}', Recognized '{rec_digit_ex}', Dist: {dist_ex:.2f}")
    results['example_recognition'] = example_rec_data

    print("\nVisualizing DTW paths for selected digit pairs (your recordings)...")
    s_label1, s_label2 = "six", "six"
    results['same_digit_label_1'], results['same_digit_label_2'] = s_label1, s_label2
    mfcc_s1 = test1_mfcc.get(s_label1)
    mfcc_s2 = test2_mfcc.get(s_label2)
    if mfcc_s1 is not None and mfcc_s2 is not None:
        plot_path_s = plotDTWpath(mfcc_s1, mfcc_s2,
                                title_prefix=f"Part (b) Same ({s_label1}): ", save_path=os.path.join("plots", f"part_b_same_{s_label1}.png"))
        results['same_digit_plot_path'] = plot_path_s if plot_path_s and os.path.exists(plot_path_s) else None
    else:
        print(f"Skipping same digit plot: MFCCs for '{s_label1}' or '{s_label2}' not available.")
    
    d_labelA, d_labelB = "six", "four"
    results['diff_digit_label_a'], results['diff_digit_label_b'] = d_labelA, d_labelB
    mfcc_dA = test1_mfcc.get(d_labelA)
    mfcc_dB = train_mfcc.get(d_labelB)
    if mfcc_dA is not None and mfcc_dB is not None:
        plot_path_d = plotDTWpath(mfcc_dA, mfcc_dB,
                                title_prefix=f"Part (b) Diff ({d_labelA} vs {d_labelB}): ", save_path=os.path.join("plots",f"part_b_diff_{d_labelA}_{d_labelB}.png"))
        results['diff_digit_plot_path'] = plot_path_d if plot_path_d and os.path.exists(plot_path_d) else None
    else:
        print(f"Skipping different digit plot: MFCCs for '{d_labelA}' or '{d_labelB}' not available.")
        
    print("\nEvaluating performance on your recordings...")
    all_test_sets_mfcc = []
    for mfcc_set in [test1_mfcc, test2_mfcc, test3_mfcc]:
        if mfcc_set and any(v is not None for v in mfcc_set.values()):
            all_test_sets_mfcc.append(mfcc_set)

    if all_test_sets_mfcc:
        accuracy, confusion_mat_arr = calculate_accuracy_and_confusion_matrix(
            train_mfcc, all_test_sets_mfcc, config.DIGITS_ORDERED
        )
        results['accuracy'] = accuracy
        results['confusion_matrix_array'] = confusion_mat_arr.tolist()
        print(f"\nOverall Accuracy on Your Recordings: {accuracy*100:.2f}%")
        
        cm_plot_path = plot_confusion_matrix(confusion_mat_arr, config.DIGITS_ORDERED, accuracy, save_path=os.path.join("plots","part_b_cm.png"))
        results['confusion_matrix_plot_path'] = cm_plot_path if cm_plot_path and os.path.exists(cm_plot_path) else None
    else:
        print("No valid test MFCC sets for Part B evaluation.")
        results['accuracy'] = "N/A (No valid test data)"
        results['confusion_matrix_array'] = np.zeros((len(config.DIGITS_ORDERED), len(config.DIGITS_ORDERED))).tolist()
        results['confusion_matrix_plot_path'] = None
        
    REPORT_DATA["part_b_results"] = results
    print("--- Part (b) Finished ---")

def part_c():
    print("\n--- Running Part (c): Evaluation on TDIGITS ---")
    results = {}

    if not os.path.exists(config.TDIGITS_BASE_PATH) or not os.listdir(config.TDIGITS_BASE_PATH):
        error_msg = f"TDIGITS data not found or directory empty at {config.TDIGITS_BASE_PATH}"
        print(f"ERROR: {error_msg} Skipping Part (c).")
        results["error"] = error_msg
        REPORT_DATA["part_c_results"] = results
        return

    print("\nPart (c).1: Same-Speaker Evaluation (TDIGITS)")
    ref_speaker_s = "jackson" 
    results['ref_speaker_same'] = ref_speaker_s
    tdigits_ref_s = load_tdigits_reference_mfcc(config.TDIGITS_BASE_PATH, ref_speaker_s, config.TDIGITS_DIGITS_STR, 
                                                config.TDIGITS_REFERENCE_REPETITION, config.MFCC_PARAMS)
    if tdigits_ref_s and not all(v is None for v in tdigits_ref_s.values()):
        acc_s, _, _ = evaluate_on_tdigits(tdigits_ref_s, ref_speaker_s, config.TDIGITS_DIGITS_STR, config.TDIGITS_BASE_PATH,
                                          num_test_repetitions=config.TDIGITS_NUM_TEST_REPETITIONS,
                                          reference_repetition_id=config.TDIGITS_REFERENCE_REPETITION,
                                          reference_speaker_id=ref_speaker_s, mfcc_params=config.MFCC_PARAMS)
        results['same_speaker_accuracy'] = acc_s
    else:
        print(f"Could not load reference set for TDIGITS speaker '{ref_speaker_s}'. Evaluation might be affected.")
        results['same_speaker_accuracy'] = "N/A (Ref data error)"

    print("\nPart (c).2: Cross-Speaker Evaluation (TDIGITS)")
    ref_speaker_c = "jackson" 
    results['ref_speaker_cross'] = ref_speaker_c
    tdigits_ref_c = tdigits_ref_s # Reuse Jackson's reference
    
    cross_accuracies = {}
    overall_cross_correct = 0
    overall_cross_tested = 0
    if tdigits_ref_c and not all(v is None for v in tdigits_ref_c.values()):
        for test_speaker in config.TDIGITS_SPEAKERS:
            if test_speaker == ref_speaker_c: continue
            acc_c, correct_c, tested_c = evaluate_on_tdigits(tdigits_ref_c, test_speaker, config.TDIGITS_DIGITS_STR, config.TDIGITS_BASE_PATH,
                                                             num_test_repetitions=config.TDIGITS_NUM_TEST_REPETITIONS,
                                                             reference_speaker_id=ref_speaker_c, mfcc_params=config.MFCC_PARAMS)
            cross_accuracies[test_speaker] = acc_c
            overall_cross_correct += correct_c
            overall_cross_tested += tested_c
        results['cross_speaker_accuracies'] = cross_accuracies
        if overall_cross_tested > 0:
            results['overall_cross_speaker_accuracy'] = overall_cross_correct / overall_cross_tested
        else:
            print("No cross-speaker tests were successfully performed (all_cross_tested = 0).")
            results['overall_cross_speaker_accuracy'] = "N/A (No test data)"
    else:
        print(f"Could not load/use reference set for TDIGITS speaker '{ref_speaker_c}' for cross-speaker eval.")
        results['cross_speaker_accuracies'] = {}
        results['overall_cross_speaker_accuracy'] = "N/A (Ref data error)"
        
    results['suggestions'] = [
        "More Robust Reference Templates (e.g., averaging, multi-speaker).",
        "Advanced Feature Normalization (e.g., Vocal Tract Length Normalization - VTLN).",
        "Improved Distance/Similarity Measures (e.g., Hidden Markov Models - HMMs).",
        "Speaker Adaptation Techniques (e.g., Maximum A Posteriori - MAP, i-vectors).",
        "Data Augmentation (adding noise, pitch/speed variations to training data)."
    ]
    REPORT_DATA["part_c_results"] = results
    print("--- Part (c) Finished ---")

if __name__ == "__main__":
    print("Starting Isolated Digit Recognition Assignment Script...")
    ensure_data_dirs_exist()

    if config.RUN_PART_A:
        part_a()
    if config.RUN_PART_B:
        part_b()
    if config.RUN_PART_C:
        part_c()
    
    generate_html_report(REPORT_DATA)
    
    print("\nAssignment Script Finished.")
    print("HTML report generated as 'assignment_report.html'.")
    print("Plots saved in 'plots/' directory.")