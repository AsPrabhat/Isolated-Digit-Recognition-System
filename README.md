<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Status-Completed-green?style=for-the-badge" alt="Status">
  <!-- Add other relevant badges if you have them, e.g., license -->
</div>

# Isolated Digit Recognition using Dynamic Time Warping (DTW)

## 🎙️ Project Overview

This project implements an **Isolated Digit Recognition** system from scratch using the **Dynamic Time Warping (DTW)** algorithm. The primary goal is to recognize spoken digits (0-9) by comparing their Mel Frequency Cepstral Coefficient (MFCC) features against a set of reference templates. The project explores the effectiveness of DTW for this task, evaluates its performance on user-recorded audio and a standard dataset (TDIGITS), and discusses potential improvements.

The system is structured into modular Python scripts for clarity and reusability, culminating in an automated HTML report summarizing the findings and visualizations.

---

## ✨ Key Features

*   **DTW from Scratch:** Core DTW algorithm implemented to find optimal alignment and distance between sequences.
*   **MFCC Feature Extraction:** Utilizes `speechpy` for extracting robust acoustic features (MFCCs) from audio signals.
*   **Isolated Digit Recognition:** Classifies spoken digits based on minimum DTW distance to reference templates.
*   **Comprehensive Evaluation:**
    *   Performance analysis on user-recorded audio data.
    *   Performance analysis on a subset of the **TDIGITS** dataset.
    *   Same-speaker vs. Cross-speaker evaluation to highlight speaker variability challenges.
*   **Automated HTML Reporting:** Generates a well-formatted HTML report with results, plots, and observations using Jinja2 templating.
*   **Modular Code Structure:** Organized into separate Python modules for core logic, audio processing, recognition, evaluation, and configuration.
*   **Visualizations:**
    *   DTW alignment paths on local and accumulated cost matrices.
    *   Confusion matrices for performance assessment.

---

## 📂 Project Structure

```
isolated_digit_recognition_project/
├── main_assignment.py      # Main script to orchestrate the assignment parts
├── dtw_core.py             # DTW algorithm implementation and plotting
├── audio_processing.py     # MFCC extraction logic
├── recognition_system.py   # Isolated digit recognition algorithm
├── evaluation.py           # Accuracy, confusion matrix, TDIGITS evaluation
├── html_reporter.py        # Generates the HTML report
├── report_template.html    # Jinja2 template for the HTML report
├── config.py               # Configuration for paths, parameters, run control
├── data/                   # Directory for YOUR recorded audio files
│   ├── zero_1.wav
│   ├── ...
│   └── nine_4.wav
├── TDIGITS_subset/         # Directory for the TDIGITS dataset
│   ├── jackson/
│   ├── nicolas/
│   ├── theo/
│   └── yweweler/
├── README.md               # This file
├── .gitignore
└── requirements.txt        # Python dependencies
```


---

## 🛠️ Setup & Installation

1.  **Prerequisites:**
    *   Python 3.9 or higher.
    *   `pip` (Python package installer).

2.  **Clone the Repository (Optional):**
    ```bash
    git clone https://github.com/AsPrabhat/Isolated-Digit-Recognition-System
    cd Isolated-Digit-Recognition-System
    ```
    Otherwise, ensure all project files are in a single directory.

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source ./venv/bin/activate`

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install `numpy`, `matplotlib`, `scipy`, `speechpy`, `seaborn`, and `Jinja2`.

5.  **Prepare Audio Data:**

    *   **Your Recordings (for Part B):**
        *   Create a directory named `data/` in the project root.
        *   Record your voice saying each digit (0-9). (Added my voice recording for reference)
            *   **Training (Reference) Set:** One recording per digit, named `zero_1.wav`, `one_1.wav`, ..., `nine_1.wav`.
            *   **Test Sets (3 repetitions):**
                *   `zero_2.wav`, ..., `nine_2.wav`
                *   `zero_3.wav`, ..., `nine_3.wav`
                *   `zero_4.wav`, ..., `nine_4.wav`
        *   Ensure files are in `.wav` format (e.g., 16-bit PCM, 16000Hz, mono).

    *   **TDIGITS Dataset (for Part C):**
        *   Contains the voice recording of 4 different person as dataset
        *   The path is configured in `config.py` (`TDIGITS_BASE_PATH`).

---

## 🚀 Running the Project

1.  **Configure Run (Optional):**
    Open `config.py` to select which parts of the assignment to execute by setting `RUN_PART_A`, `RUN_PART_B`, and `RUN_PART_C` to `True` or `False`.

2.  **Execute the Main Script:**
    From the project's root directory, run:
    ```bash
    python main_assignment.py
    ```

3.  **View Results:**
    *   Console output will show progress and summary results for each part.
    *   Generated plots will be saved in the `plots/` directory.
    *   A comprehensive **`assignment_report.html`** file will be created in the project root. Open this file in a web browser to view a structured report with all results and visualizations.

---

## 📈 Expected Output & Observations

The script will perform the following and generate relevant outputs:

*   **Part (a): Basic DTW**
    *   Calculates and displays the DTW distance for example sequences.
    *   Generates and saves a plot visualizing the DTW path.
*   **Part (b): Recognition on Your Recordings**
    *   Extracts MFCCs from your `.wav` files in the `data/` directory.
    *   Performs an example recognition.
    *   Visualizes DTW paths for same-digit and different-digit pairs.
    *   Calculates and displays overall accuracy.
    *   Generates and saves a confusion matrix plot.
*   **Part (c): Evaluation on TDIGITS**
    *   Extracts MFCCs from the `TDIGITS_subset/` directory.
    *   Calculates and displays same-speaker accuracy.
    *   Calculates and displays cross-speaker accuracies.
    *   Lists suggestions for system improvement.

**Detailed observations for each part should be inferred from the generated `assignment_report.html` and console output.**

---

## 💡 Potential Improvements & Future Work

*   **More Robust Reference Templates:** Instead of single utterances, use techniques like averaging MFCCs from multiple recordings or creating statistical models (e.g., GMMs) per digit.
*   **Speaker-Independent Training:** Incorporate reference templates from multiple speakers.
*   **Advanced Feature Normalization:** Explore Vocal Tract Length Normalization (VTLN) or other speaker normalization techniques.
*   **Alternative Similarity Measures:** Investigate Hidden Markov Models (HMMs), which are standard in ASR and can model temporal variations more robustly.
*   **Speaker Adaptation:** Implement techniques (e.g., MAP, i-vectors) if a small amount of data from a new speaker is available.
*   **Data Augmentation:** Increase training data diversity by adding noise, reverberation, or applying pitch/speed variations.

---

## 📞 Contact

Prabhhat Pensalwar

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/prabhat-pensalwar-2ab7a5330/)  [![GitHub](https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github)](https://github.com/AsPrabhat)  [![Email](https://img.shields.io/badge/Email-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:prabhatworkspace@gmail.com)


---

<div align="center">
  <em>Thank you for reviewing this project!</em>
</div>