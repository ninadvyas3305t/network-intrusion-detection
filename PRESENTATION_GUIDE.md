# College Presentation & Operating Guide: AI-Based Network Intrusion Detection System

**Author / Presenter:** 3rd-Year AI & Data Science Engineering Student  
**Project:** AI-Based Network Intrusion Detection System (NIDS)  
**Primary Dataset:** UNSW-NB15 Benchmark (Australian Centre for Cyber Security)  
**Primary Model:** XGBoost Classifier (with Random Forest and Logistic Regression comparisons)  
**Explainability Engine:** SHAP TreeExplainer  
**Interface:** Streamlit Dashboard  

---

## 1. Project Overview in Simple Language

### What Problem Are We Solving?
Computer networks face thousands of connection attempts every minute. Some are legitimate users browsing web pages or sending emails, but others are malicious adversaries attempting to hack systems, flood servers with Denial-of-Service (DoS) attacks, or steal data. 

Traditionally, firewalls use fixed rules (called **signatures**) to catch known attacks. But if a hacker modifies the attack slightly or uses a brand-new technique, rule-based systems fail. 

Our project uses **Machine Learning** to inspect connection flows. Instead of matching fixed strings, our model learns statistical behavioral patterns (such as packet transmission rate, byte sizes, and connection states) to automatically detect:
1. **Whether traffic is Normal or Malicious** (Binary Classification).
2. **Which exact attack category it belongs to** (Multiclass Classification across 10 categories).

### What Does Our System Actually Do?
1. It ingests network flow metadata (42 flow features like duration, protocol, service, packet count, Time-To-Live).
2. It scales numerical values and encodes categorical attributes without data leakage.
3. It passes the transformed flow through our trained **XGBoost** model.
4. It outputs an immediate decision (**Normal** vs. **Attack**) along with a probability score (e.g., 94.2% Attack probability).
5. If classified as an attack, it predicts the threat category (e.g., DoS, Exploits, Reconnaissance, Worms).
6. It uses **SHAP (SHapley Additive exPlanations)** to explain *why* the decision was made (e.g., "Source TTL was unusually high (+1.06 impact) and connection rate was abnormally dense").

---

## 2. Complete Architecture & File Responsibilities

### The Complete End-to-End Pipeline
$$\text{Dataset} \longrightarrow \text{Cleaning} \longrightarrow \text{Feature Engineering} \longrightarrow \text{Training} \longrightarrow \text{Thresholding} \longrightarrow \text{SHAP} \longrightarrow \text{Streamlit UI}$$

| Stage | Responsible File | Exact Functionality |
| :--- | :--- | :--- |
| **Dataset Acquisition** | `src/download_dataset.py` | Automatically downloads the official partitioned UNSW-NB15 CSV files (`UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`) into `data/raw/`. |
| **Data Cleaning** | `src/data_preprocessing.py` | Eliminates index leakage (`id` column), strips string whitespaces, normalizes lowercase casing, replaces placeholder `'-'` in service with `'none'`, and saves `data/processed/train_cleaned.csv` and `test_cleaned.csv`. |
| **Feature Engineering** | `src/feature_engineering.py` | Constructs a Scikit-learn `ColumnTransformer` (`StandardScaler` for 39 numerical features + `OneHotEncoder` with `handle_unknown='ignore'` for 3 categorical features: `proto`, `service`, `state`). Fits strictly on training data to prevent leakage. Serializes `models/preprocessor.joblib`, `models/label_encoder.joblib`, and `data/processed/train_features.npz`. |
| **Model Training & Benchmarking** | `src/train.py` | Trains Logistic Regression (baseline), Random Forest, and XGBoost on both Binary and Multiclass tasks. Uses `time.perf_counter()` and `tracemalloc` to measure exact training time and peak memory consumption. Generates confusion matrices and outputs `models/training_results.json`. |
| **Evaluation Metrics** | `src/evaluate.py` | Calculates Accuracy, Precision, Recall, $F_1$-Score, Macro-$F_1$, Weighted-$F_1$, ROC-AUC, and per-sample latency. |
| **Threshold Tuning** | `src/threshold_analysis.py` | Evaluates decision threshold trade-offs ($0.10 \dots 0.90$) on a 20% validation split of training data to prevent test leakage. Analyzes False Positives vs. False Negatives and outputs `models/threshold_analysis.json`. |
| **Model Explainability** | `src/explainability.py` | Wraps XGBoost with `shap.TreeExplainer`. Generates global feature importance rankings (`models/shap_summary.json`) and computes local instance explanations for single flow analysis. |
| **Inference Engine** | `src/predict.py` | Unified prediction engine (`IntrusionPredictor`). Loads preprocessor, encoders, and models. Handles column validation, batch CSV inference, manual single-flow inputs, probability outputs, and SHAP integration. |
| **Web Dashboard** | `app/app.py` | 5-section Streamlit application. Caches models with `@st.cache_resource`, renders interactive charts, handles CSV uploads, and provides preset flow buttons. |
| **Test Suite** | `tests/` (5 files) | Pytest suite containing 20 tests verifying preprocessing, pipeline compatibility, model loading, thresholding, SHAP, and Streamlit assets. |

---

## 3. How to Run the Website on Your Mac

### Step 1: Open Terminal and Navigate to Project
Open your Terminal application and enter:
```bash
cd "/Users/ninadvyas/Documents/network intrusion detection"
```

### Step 2: Activate the Virtual Environment
```bash
source .venv/bin/activate
```
*(Your terminal prompt should now show `(.venv)` at the beginning of the line).*

### Step 3: Run the Test Suite (Sanity Check)
```bash
pytest -v
```
*(You will see all 20 tests pass in approximately 11 seconds).*

### Step 4: Launch the Streamlit Dashboard
```bash
streamlit run app/app.py
```

### Step 5: Open in Your Browser
Streamlit will automatically open your browser at:
`http://localhost:8501`

### Step 6: How to Stop the App
When you are done with the demonstration, go back to Terminal and press:
`Ctrl + C`

---

## 4. Explanation of All Five Dashboard Sections

### Section 1: Executive Dashboard
- **What the teacher sees**: 5 high-level metric cards (Total Records: 257,673, Test Accuracy: 87.34%, Attack Recall: 98.65%, F1-Score: 0.8956, ROC-AUC: 0.9851), a benchmark notice, a donut chart of Normal vs. Attack test distribution, and a Mermaid pipeline diagram.
- **Where the numbers come from**: These are empirical, measured evaluation results from evaluating our trained XGBoost model on the 82,332 test samples stored in `models/training_results.json`.
- **Benchmark vs. Live distinction**: The dashboard metrics represent the **fixed benchmark performance on the official test partition**, whereas Section 3 performs **live predictions on newly provided data**.
- **What to say**:
  > *"Here on the Executive Dashboard, we see the high-level evaluation metrics on the official UNSW-NB15 benchmark test partition of 82,332 samples. Our primary XGBoost model achieves 87.34% accuracy and an attack recall of 98.65%, meaning it catches nearly 99 out of every 100 intrusions. All metrics displayed here come from actual hardware measurements."*

### Section 2: Dataset & EDA
- **What the teacher sees**: 3 tabs: Class & Attack Taxonomy, Correlation & Collinearity Study, and Feature Distributions.
- **Key takeaways to highlight**:
  1. **Long-Tailed Imbalance**: Dominant classes (Normal: 36%, Generic: 23%, Exploits: 17%) make up over 76% of traffic, while rare threats like Worms (174 total rows) and Shellcode (1,511 rows) make up less than 1%.
  2. **Collinearity Rationale**: Show the correlation heatmap. Point out that pairs like `sbytes` $\leftrightarrow$ `sloss` ($r=0.9961$) and `tcprtt` $\leftrightarrow$ `synack` ($r=0.9495$) correlate due to physical networking protocols. Explain that decision trees naturally handle collinearity, so keeping both packet counts and byte volumes allows the model to learn average packet size distinctions.
- **What to say**:
  > *"In this section, we explore the UNSW-NB15 dataset. Notice the realistic long-tailed class distribution. Rather than blindly dropping correlated features, we analyzed their networking domain meaning. Tree-based models evaluate orthogonal splits, so keeping both packet and byte counts enables the model to detect small-packet DoS attacks versus large-payload exfiltration."*

### Section 3: Intrusion Detection (Live Inference)
- **What the teacher sees**: Two modes: Batch CSV Traffic Analysis and Single Network Flow Simulator.
- **How Batch CSV Works**:
  - The teacher or student uploads a CSV.
  - The pipeline validates that all required flow features are present.
  - If test columns (`id`, `label`, `attack_cat`) are present, the pipeline automatically strips them without crashing.
  - It outputs total records, Normal vs. Attack breakdown, predicted attack categories, and allows CSV download.
  - **Sample Batch Button**: Click `📥 Download Sample Test Batch (50 Mixed Flows)` to get an authentic 50-row CSV from the test partition, then drag-and-drop it into the uploader!
- **How the Single Flow Presets Work**:
  - **Important Code Reality**: The 4 preset buttons (`🟢 Normal Web Flow`, `🔴 Reconnaissance / Port Scan`, `🔴 DoS Flooding Attack`, `🔴 Web Exploit`) are **manually constructed domain-representative flow profiles** defined in `app/app.py` lines 444–470 to simulate typical attack behaviors. Meanwhile, the 50-sample batch CSV contains **real recorded dataset flows**.
  - Clicking a preset populates the form fields.
  - Clicking `🚀 Analyze Network Flow` runs `predict_single()` and displays the instant prediction, confidence, and local SHAP factor breakdown.
- **What the Decision Threshold Slider Does**:
  - Located in the sidebar (0.10 to 0.90, default 0.50).
  - Shifting it below 0.50 increases attack sensitivity (paranoid mode, catches stealthy attacks, more false alarms).
  - Shifting it above 0.50 requires higher confirmation (conservative mode, fewer false alarms).

### Section 4: Model Performance & Benchmarking
- **What the teacher sees**: Multi-model comparison table (Logistic Regression vs. Random Forest vs. XGBoost), confusion matrices, multiclass benchmarks, and threshold trade-off curves.
- **Why XGBoost over Random Forest**:
  - Random Forest had slightly higher attack recall at default threshold (98.94% vs. 98.65%, missing 482 vs. 611 attacks).
  - However, XGBoost achieved higher precision (82.00% vs. 80.92%), higher overall F1 (0.8956 vs. 0.8903), higher ROC-AUC (0.9851 vs. 0.9824), and trained in **0.93 seconds** using **0.67 MB RAM** (vs. 2.22s and 101.5 MB RAM for Random Forest).
  - Setting XGBoost's threshold to 0.40 yields **99.52% attack recall** (only 219 missed attacks), readily outperforming Random Forest.
- **Macro-F1 vs. Weighted-F1**:
  - Weighted-F1 is high (0.7792) because majority classes perform well.
  - Macro-F1 is lower (0.5045) because it penalizes low performance on difficult stealthy minority classes (Analysis, Backdoor).

### Section 5: Model Explainability (SHAP)
- **What the teacher sees**: Global Top 15 SHAP bar chart, feature descriptions table, and local attribution mathematical formulas.
- **What Top Features Mean**:
  - `sttl` (Source TTL): Modulated TTL indicates operating system fingerprinting or network hops.
  - `smean` (Mean packet size): Attack probes have distinct byte size footprints compared to standard web browsing.
  - `ct_srv_dst` (Connection count to destination service): High counts expose port scanning and flooding.
- **Viva Disclaimer (Attribution vs. Causation)**:
  - SHAP shows feature mathematical attribution within the learned model manifold; it does not prove physical forensic causation.

---

## 5. Live 5–7 Minute Teacher Demonstration Script

| Time | Action / Click | What to Point at | What to Say | Fallback if Issue Occurs |
| :--- | :--- | :--- | :--- | :--- |
| **0:00 - 1:00** | Open Dashboard page (`http://localhost:8501`). | Point to Title and 5 Metric Cards at top. | *"Good morning, Professor. This is my project: an AI-Based Network Intrusion Detection System. It solves the limitation of signature-based firewalls by using machine learning to detect zero-day and anomalous traffic based on flow statistics. Here on the Executive Dashboard, we see our primary XGBoost model achieves 87.34% accuracy and 98.65% attack recall on the official UNSW-NB15 test set."* | Refresh browser if page does not load immediately. |
| **1:00 - 2:00** | Click **"Dataset & EDA"** in sidebar. Click **"🔗 Correlation & Collinearity Study"** tab. | Point to Correlation Heatmap and table. | *"Our dataset is UNSW-NB15 with 257,673 records. During EDA, we identified 16 feature pairs with correlation above 0.90, like source bytes and packet loss. Rather than blindly dropping them, we retained them because tree-based models evaluate orthogonal splits, allowing the model to distinguish small-packet flood DoS from large payload exfiltration."* | Click on Tab 1 if image rendering takes a second. |
| **2:00 - 3:30** | Click **"Intrusion Detection"** in sidebar. Click **"📥 Download Sample Test Batch"**. Upload it into the file uploader. | Point to "Total Flows: 50", "Attack Flows: 49", and the Attack Category bar chart. | *"In the live Intrusion Detection Center, I can upload a batch CSV. I'll download this authentic 50-flow sample and upload it. The pipeline validates all 42 required features, runs our preprocessor, and classifies the traffic in milliseconds. It separates Normal from Attack and breaks down the attack categories—such as Exploits, Fuzzers, and DoS."* | If upload is slow, show the preview table directly. |
| **3:30 - 4:30** | Switch radio to **"🔍 Single Network Flow Simulator"**. Click **"🔴 Reconnaissance / Port Scan"** preset, then click **"🚀 Analyze Network Flow"**. | Point to red "INTRUSION DETECTED" card and the local SHAP bar chart below it. | *"Next, we can simulate individual connections. I'll load a Port Scan preset and click Analyze. The model flags this as an Attack with over 90% confidence. Below, the local SHAP explanation shows exactly why: Source TTL and connection counts pushed the score strongly into the attack category."* | Click "Analyze Network Flow" again if preset didn't update. |
| **4:30 - 5:30** | Click **"Model Performance"** in sidebar. | Point to side-by-side comparison table (LR vs. RF vs. XGBoost). | *"In this section, we compare our three models. Logistic Regression provides our linear baseline at 80.95% accuracy. Random Forest achieved 98.94% attack recall, but XGBoost achieved higher precision, higher F1 of 0.8956, and higher ROC-AUC of 0.9851, while training in just 0.93 seconds."* | Point to the saved confusion matrix figures. |
| **5:30 - 6:30** | In sidebar, change threshold slider from **0.50 to 0.40**. | Point to the sidebar badge changing to "High Sensitivity". | *"Finally, our sidebar includes a dynamic decision threshold slider. If an organization wants zero-tolerance security, lowering the threshold to 0.40 raises attack recall to 99.52%, missing only 219 attacks out of 45,332."* | Reset slider to 0.50. |
| **6:30 - 7:00** | Click **"Explainability (SHAP)"** in sidebar. Conclude presentation. | Point to Top 15 SHAP bar chart and the viva disclaimer. | *"In conclusion, our system provides high-speed, leak-free network intrusion detection with full XAI transparency through SHAP. Thank you, Professor. I am ready for your questions."* | Ready for viva questions. |

---

## 6. Our Actual ML Results Explained

All metrics below are **empirically measured** and recorded in `models/training_results.json`:

### Binary Classification (Normal vs. Attack on 82,332 Test Samples)
- **Accuracy = 87.34%**: Out of 82,332 test flows, the model correctly classified 71,906 flows.
- **Precision = 82.00%**: When the model flags a connection as an Attack, it is correct 82% of the time (18% are false alarms).
- **Recall = 98.65%**: Out of 45,332 actual attack flows in the test set, the model successfully caught **44,721 attacks**, missing only 611 (1.35% breach rate).
- **$F_1$-Score = 0.8956**: The harmonic mean of precision and recall, balancing false alarms against missed attacks.
- **ROC-AUC = 0.9851**: Near-perfect area under the ROC curve, proving excellent rank discrimination across all probability thresholds.
- **Training Time = 0.93 seconds**: Fast training enabled by Apple Silicon M-series hardware and XGBoost's histogram-based tree building.
- **Peak RAM = 0.67 MB**: Extremely lightweight memory footprint.
- **Inference Latency = 0.00022 ms / flow**: Can process over 4 million flow records per second.

### Confusion Matrix Breakdown (Actual Numbers)
- **True Negatives (TN)** = **27,185**: Legitimate flows correctly marked as Normal.
- **False Positives (FP)** = **9,815**: Legitimate flows incorrectly flagged as Attack (alert fatigue).
- **False Negatives (FN)** = **611**: Actual attacks missed by the model (security breach).
- **True Positives (TP)** = **44,721**: Actual attacks successfully detected.

### Multiclass Results (10 Categories)
- **Overall Accuracy** = 76.79%
- **Weighted-$F_1$** = 0.7792 (high because majority classes Generic, Normal, Exploits perform well).
- **Macro-$F_1$** = 0.5045 (unweighted arithmetic mean, penalizing difficult minority classes).
- **Rare Class Performance**:
  - **Worms (44 test samples)**: Precision = 60.0%, Recall = 47.7%, $F_1$ = **0.5316** (caught 21 of 44 extreme-rare worms).
  - **Shellcode (378 test samples)**: Precision = 32.3%, Recall = **78.8%**, $F_1$ = **0.4585** (caught 298 of 378 shellcodes).
  - **Analysis & Backdoor**: Hardest classes ($F_1 < 0.10$) due to behavioral overlap with legitimate encrypted web traffic.

---

## 7. Likely Viva Questions & Answers (Student Cheat Sheet)

#### Q1: Why did you choose this project?
- **Short Answer:** To address the limitations of traditional rule-based firewalls by building an AI system that detects zero-day attacks and encrypted flow anomalies with explainable AI.
- **Deeper Follow-up:** Signature-based systems fail against novel attacks and cannot inspect encrypted traffic without costly SSL decryption. Flow-based ML analyzes packet metadata to detect anomalies at line rate.

#### Q2: What is the difference between an IDS and an IPS?
- **Short Answer:** An IDS (Intrusion Detection System) *monitors and alerts* on suspicious activity; an IPS (Intrusion Prevention System) *actively blocks or drops* the malicious traffic in real time.
- **Deeper Follow-up:** Our software functions as an IDS intelligence engine. In a production pipeline, its prediction API would feed signals to an IPS firewall to trigger automated IP blacklisting.

#### Q3: Why did you choose the UNSW-NB15 dataset over NSL-KDD and CICIDS2017?
- **Short Answer:** NSL-KDD is from 1998 (outdated protocols), while CICIDS2017 is over 3 GB with severe data bugs (infinite values, 288k duplicates). UNSW-NB15 offers modern CVE attacks with clean data quality and manageable compute.
- **Deeper Follow-up:** UNSW-NB15 was created in 2015 using IXIA PerfectStorm. It includes 9 modern attack categories and 49 features reflecting modern TCP/IP behaviors.

#### Q4: Why use Machine Learning instead of fixed signatures?
- **Short Answer:** Machine learning generalizes from patterns, enabling detection of mutated malware, zero-day vulnerabilities, and multi-stage attacks that have no existing signature.

#### Q5: What is the difference between Binary and Multiclass classification in your project?
- **Short Answer:** Binary classification acts as the frontline filter (Normal vs. Attack); Multiclass classification provides incident triage by attributing the attack to 10 specific categories.

#### Q6: Why did you compare three models?
- **Short Answer:** To follow the scientific ML lifecycle: Logistic Regression sets the linear baseline, Random Forest provides a bagged ensemble benchmark, and XGBoost provides boosted performance.
- **Deeper Follow-up:** Comparing models proves that non-linear tree ensembles significantly outperform linear decision boundaries on network traffic.

#### Q7: Why did you select XGBoost as the primary model if Random Forest had higher recall?
- **Short Answer:** XGBoost achieved higher precision (82% vs 80.9%), higher F1 (0.8956 vs 0.8903), higher ROC-AUC (0.9851 vs 0.9824), and trained 2.4x faster with 150x less memory. Lowering XGBoost's threshold to 0.40 easily surpasses Random Forest's recall.

#### Q8: Why is accuracy insufficient for network intrusion detection?
- **Short Answer:** Because of class imbalance. A naive model predicting 100% Normal traffic could achieve high accuracy while missing every single cyberattack. Recall and F1 are essential.

#### Q9: What is a Confusion Matrix?
- **Short Answer:** A $2 \times 2$ grid showing True Negatives, False Positives (false alarms), False Negatives (missed intrusions), and True Positives (caught intrusions).

#### Q10: What is ROC-AUC?
- **Short Answer:** Receiver Operating Characteristic Area Under the Curve. It measures how well the model separates Normal from Attack across all possible decision thresholds ($1.0$ is perfect, $0.5$ is random). Our score is **0.9851**.

#### Q11: What is Class Imbalance and how did you handle it?
- **Short Answer:** Class imbalance occurs when one class heavily outnumbers others (Normal: 93,000 vs. Worms: 174). We used stratified splitting, evaluated Macro-F1, and used balanced class weighting.

#### Q12: What is Overfitting and how did you prevent it?
- **Short Answer:** Overfitting is when a model memorizes training noise and fails on unseen data. We prevented it using depth constraints (`max_depth=6`), sub-sampling (`0.8`), and strictly testing on the held-out test partition.

#### Q13: What is Data Leakage and how did you prevent it?
- **Short Answer:** Data leakage happens when test set information inadvertently influences training. We dropped the `id` column and fitted all scalers and encoders **strictly on `X_train`**.

#### Q14: Why did you use StandardScaler?
- **Short Answer:** Features like bytes range into millions while duration is in fractions of a second. StandardScaler normalizes continuous features to $\mu=0, \sigma=1$, ensuring linear models and distance calculations converge properly.

#### Q15: Why did you use OneHotEncoder with handle_unknown='ignore'?
- **Short Answer:** It converts categorical strings (`proto`, `service`, `state`) into binary indicator columns. `handle_unknown='ignore'` guarantees that if a new protocol appears during live inference, the app won't crash.

#### Q16: What is SHAP?
- **Short Answer:** SHapley Additive exPlanations, a game-theoretic approach that explains machine learning predictions by quantifying the exact marginal contribution of each feature.

#### Q17: What does a positive vs. negative SHAP value mean in your app?
- **Short Answer:** A positive SHAP value pushes the prediction toward **Attack**; a negative SHAP value mitigates the prediction toward **Normal**.

#### Q18: Does SHAP prove that a feature caused an attack?
- **Short Answer:** No. SHAP measures mathematical feature attribution within the model's learned manifold; it does not constitute forensic causal proof.

#### Q19: What is the Decision Threshold?
- **Short Answer:** The probability cutoff (default $0.50$) at which a flow is classified as Attack. Lowering it increases sensitivity; raising it increases precision.

#### Q20: Why keep 0.50 as the default threshold?
- **Short Answer:** On a held-out validation split of training data, $F_1$-score peaked at $0.50$ ($F_1 = 0.9669$). It maintains an optimal trade-off of 98.65% recall with manageable false positives.

#### Q21: Does the project inspect live raw network packets from Wi-Fi/Ethernet?
- **Short Answer:** No, this is a software-level flow analyzer operating on extracted flow statistics (CSV/batch/manual input). Connecting to live interfaces via PyShark/Zeek is part of our future scope.

#### Q22: What are the main limitations of the system?
- **Short Answer:** It analyzes flow metadata rather than encrypted packet payloads, and it struggles with stealthy minority attacks like `Analysis` and `Backdoor` ($F_1 < 0.10$) that mimic normal web traffic.

---

## 8. Troubleshooting Guide

| Issue | Cause | Fix Command / Solution |
| :--- | :--- | :--- |
| **Port 8501 is already in use** | A previous Streamlit instance is still running in the background. | Run Streamlit on a different port: `streamlit run app/app.py --server.port 8502` |
| **`ModuleNotFoundError: No module named 'src'`** | Virtual environment is inactive or pythonpath is unset. | Activate venv: `source .venv/bin/activate` and ensure you launch from the project root. |
| **`Address already in use` error** | Kill the orphaned process occupying port 8501. | In Terminal, run: `lsof -i :8501` then `kill -9 <PID>` |
| **Uploaded CSV shows error: "Missing required features"** | The CSV is missing one of the 42 required flow feature columns. | Click the `📥 Download Sample Test Batch` button in the app and upload that file to demonstrate valid CSV handling. |
| **Matplotlib cache permission warning** | Mac system `/Users/ninadvyas/.matplotlib` permissions. | Harmless warning; our scripts automatically set `os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"`. |

---

## 9. Final Checklist Before Demonstrating to Your Teacher

- [x] Tested terminal navigation: `cd "/Users/ninadvyas/Documents/network intrusion detection"`
- [x] Activated virtual environment: `source .venv/bin/activate`
- [x] Ran automated test suite: `pytest -v` (20 passed in ~11s)
- [x] Verified `sample_test_batch.csv` is present in `data/processed/`
- [x] Verified all serialized models exist in `models/` (`best_model_binary.joblib`, `best_model_multiclass.joblib`, `preprocessor.joblib`, `label_encoder.joblib`)
- [x] Verified Streamlit dashboard opens cleanly: `streamlit run app/app.py`
- [x] Practiced the 5-7 minute demonstration script
- [x] Reviewed the 22 viva questions in this guide and `VIVA_QA.md`
