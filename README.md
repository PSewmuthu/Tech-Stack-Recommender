# Tech-Stack-Recommender

A content-based job recommendation engine developed as a capstone project during my one-month internship at DecodeLabs. This tool leverages machine learning techniques to map user skills directly to relevant job roles, effectively solving "Choice Overload" in professional career searching.

## Core Methodology

The engine operates on the **Input-Process-Output** model:

- **Input (User State):** Captures user-provided technical skill sets.
- **Process (Similarity Logic):**
  - **Vector Mapping:** Transforms qualitative skill sets into numerical vectors.
  - **TF-IDF Weighting:** Used to penalize generic, high-frequency terms and reward highly specific technical skills.
  - **Cosine Similarity:** Calculates the angular alignment between user skill vectors and job role vectors to ensure objective, ranked results.
- **Output (Top-N List):** Generates a tailored, truncated list of the most relevant job postings.

## Key Technical Features

- **Content-Based Filtering:** The system relies on intrinsic item attributes, avoiding the "Cold Start" problem associated with collaborative filtering.
- **Python Stack:** Implemented using `pandas` for data manipulation and `scikit-learn` for TF-IDF vectorization and cosine similarity calculations.
- **Normalization:** Includes a custom pre-processing pipeline to convert skill strings into unified, underscore-separated tokens for accurate tokenization.

## Getting Started

1. Clone this repository.
2. Ensure you have the `data/all_job_post.csv` dataset in the root directory.
   - [Download/View the Dataset](https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set)
3. Run the main processing script to generate recommendations based on your skill input.

---

_Developed during the 2026 May-June Batch Internship Program at DecodeLabs._
