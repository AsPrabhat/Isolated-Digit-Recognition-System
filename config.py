# config.py

# --- General Configuration ---
DIGITS_ORDERED = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
MFCC_PARAMS = {'fft_length': 1103, 'num_cepstral': 13} # Consistent MFCC parameters

# --- Paths for Your Recorded Data (Part B) ---
# Assumes 'data' folder is in the same directory as main_assignment.py
DATA_BASE_PATH = "data/" 

# --- Paths and Config for TDIGITS Data (Part C) ---
# Assumes 'TDIGITS_subset' folder is in the same directory
TDIGITS_BASE_PATH = "TDIGITS_subset/"
TDIGITS_SPEAKERS = ["jackson", "nicolas", "theo", "yweweler"]
TDIGITS_DIGITS_STR = [str(i) for i in range(10)] # '0' through '9'
TDIGITS_REFERENCE_REPETITION = '0'
TDIGITS_NUM_TEST_REPETITIONS = 50 # Max repetitions to check per speaker/digit

# --- Control which parts of the assignment to run ---
RUN_PART_A = True
RUN_PART_B = True
RUN_PART_C = True