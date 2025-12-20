### 1\. Data Access & Download

To get the complete set of raw video files, download the compressed archive from the cloud storage link below:

- **Download Link:** [Training Videos](https://www.dropbox.com/scl/fi/daqmxyxycgr4r39lzn7gf/video.zip?rlkey=uphg8vs7n4mrdgl7bzi62y0on&e=1&dl=0)
  - **Size:** Approximately **5.3 GB** (Compressed ZIP).
  - **Format:** MP4 video files.
  - **Content:** 67 Videos of sign language gestures used to train the model.

---

### 2\. Data Mapping

We have curated a list of 31 perfectly mapped classes. These words were selected based on the final Model Classification Report, where they achieved a perfect F1-Score of 1.00 (100% Precision and 100% Recall). This ensures that users receive highly reliable, high-fidelity feedback during their learning.

The training dataset consists of 67 unique sign language folders. During the preprocessing phase, these labels were numerically sorted and mapped to 0-indexed tensors (0−66). However, the raw folder sequence on disk contains gaps (e.g., Folder 19 and Folder 33 are missing).

To address this, we implemented a reverse_label_map in the FastAPI backend. This dictionary translates the model's categorical output index back to the original KSL Folder ID. This mapping ensures the Frontend wordList remains perfectly synchronized with the internal model logic, preventing "off-by-one" errors caused by the non-sequential folder numbers.

### 3\. License & Usage

- **Data License:** The raw video data is licensed under the **CC BY-NC 4.0 International License**.
- **Source Paper:** The dataset is based on research described in the paper: [The Korean Sign Language Dataset for Action Recognition](https://www.google.com/search?q=https://link.springer.com/content/pdf/10.1007/978-3-030-37731-1_43.pdf)
