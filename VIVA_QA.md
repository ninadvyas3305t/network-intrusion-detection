# College ML Viva Preparation & Defense Guide: AI-Based Network Intrusion Detection System

This document contains concise, technically accurate answers prepared for a 3rd-year Artificial Intelligence and Data Science Engineering college project defense and viva examination.

---

### 1. Fundamentals of Intrusion Detection
#### Q1: What is a Network Intrusion Detection System (NIDS)?
**Answer:** A Network Intrusion Detection System is a security software mechanism that inspects inbound and outbound network traffic to detect unauthorized activity, malicious exploit attempts, policy violations, and anomalous flow behaviors, alerting system administrators before compromise occurs.

#### Q2: What is the difference between Signature-Based and Anomaly/ML-Based IDS?
**Answer:**
- **Signature-Based IDS** (e.g., traditional Snort rules) compares packets against a static database of known attack fingerprints. It achieves near-zero false alarms on known threats but fails completely against zero-day or polymorphic attacks.
- **ML/Anomaly-Based IDS** learns statistical patterns of normal and attack network flows, enabling detection of novel variations and stealthy multi-stage attacks without pre-existing signature definitions.

#### Q3: What is the difference between Flow-Based IDS and Deep Packet Inspection (DPI)?
**Answer:**
- **Flow-Based IDS** (this project) analyzes connection-level statistical metadata (packet count, byte volume, duration, TTL, inter-arrival time, window size) without inspecting payload text. It is lightweight, line-rate capable, and works seamlessly over end-to-end encrypted traffic (HTTPS/TLS).
- **Deep Packet Inspection (DPI)** reconstructs and decrypts the full application payload (e.g., HTTP bodies) to scan for regex patterns. DPI is computationally expensive and is increasingly rendered blind by modern TLS 1.3 encryption.

---

### 2. Dataset Selection & Data Science
#### Q4: Why was the UNSW-NB15 dataset selected?
**Answer:** Created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra, UNSW-NB15 provides a hybrid of real benign background traffic and modern synthetic attacks generated using the IXIA PerfectStorm tool. It includes 49 features reflecting modern protocols, realistic long-tailed class imbalance, and 9 modern attack categories.

#### Q5: Why not use NSL-KDD or CICIDS2017?
**Answer:**
- **Why not NSL-KDD?** NSL-KDD is based on the 1998 DARPA capture (nearly 30 years old). It lacks modern web attacks, cloud protocols, modern operating system TTL dynamics, and contemporary malware behaviors.
- **Why not CICIDS2017?** While modern, raw CICIDS2017 is over 3.1 GB (2.83M rows) with notorious data quality bugs: hundreds of thousands of `Infinity`/`NaN` entries in flow byte rates, ~288,000 duplicate rows, and single-digit attack samples (e.g., Heartbleed with only 11 rows). UNSW-NB15 provides modern threat realism without crippling data hygiene defects or excessive compute bottlenecks.

#### Q6: Why did you strictly preserve the official UNSW-NB15 train/test split?
**Answer:** The dataset creators partitioned the dataset (175,341 train rows, 82,332 test rows) across distinct capture periods to test genuine out-of-sample generalization. Randomly shuffling and recombining would leak temporal and session dependencies between train and test sets, inflating accuracy artifically and violating academic benchmarking protocols.

#### Q7: What is Data Leakage and how did you prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset influences model training. We prevented it by:
1. Immediately dropping the `id` column (row index leak).
2. Fitting all preprocessing transformers (`StandardScaler`, `OneHotEncoder`, `LabelEncoder`) **strictly on the training partition (`X_train`)**, and applying only `.transform()` on the test and inference inputs.

#### Q8: What preprocessing transformations were applied?
**Answer:**
- Numerical features (39 continuous/discrete flow statistics) were standardized using `StandardScaler` to achieve zero mean and unit variance ($\mu=0, \sigma=1$).
- Categorical features (`proto`, `service`, `state`) were encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to prevent crashes on novel protocols.
- Whitespace and casing were normalized, and placeholder `'-'` in the service column was mapped to `'none'`.

#### Q9: Why did you not delete features with high correlation (|r| ≥ 0.90)?
**Answer:** In network protocols, correlations reflect physical mechanics (e.g., `sbytes` and `sloss` correlate at $r=0.9961$ because packet loss scales with transmitted bytes; `tcprtt` and `synack` correlate at $r=0.9495$ because SYN-ACK latency is a component of RTT). Decision trees (Random Forest, XGBoost) split on orthogonal thresholds and are unaffected by collinearity. Removing them would discard valuable interaction signals (e.g., ratio of packets to bytes distinguishing small-packet DoS floods from bulk exfiltration).

---

### 3. Machine Learning Algorithms & Benchmarking
#### Q10: Why implement both Binary and Multiclass classification?
**Answer:**
- **Binary Classification (Primary)** acts as the frontline firewall filter: rapid, high-confidence determination of whether incoming traffic is Normal or Malicious.
- **Multiclass Classification (Secondary)** provides incident response triage by attributing the detected intrusion to one of 10 threat classes (Generic, Exploits, DoS, Fuzzers, Reconnaissance, Shellcode, Worms, Analysis, Backdoor).

#### Q11: What models were evaluated and why?
**Answer:**
1. **Logistic Regression (Baseline)**: Simple linear classifier regularized via L2 Ridge penalty to establish the linear separability baseline.
2. **Random Forest**: Bagged ensemble of 100 decision trees to capture non-linear feature interactions and high attack recall.
3. **XGBoost (Selected Primary Model)**: Gradient boosted trees optimizing a second-order Taylor expansion of the loss function with L1/L2 regularization to minimize false positives and maximize $F_1$.

#### Q12: Why was XGBoost chosen over Random Forest if Random Forest had slightly higher attack recall?
**Answer:**
- Random Forest achieved 98.94% attack recall vs. XGBoost's 98.65% (missing 482 vs. 611 attacks).
- However, XGBoost achieved superior Precision (82.00% vs. 80.92%), higher overall $F_1$-Score (**0.8956** vs. 0.8903), higher ROC-AUC (**0.9851** vs. 0.9824), and trained in **0.93s** with **0.67 MB RAM** (vs. 2.22s and 101.5 MB RAM for Random Forest).
- Furthermore, as shown in our threshold analysis, tuning XGBoost's decision threshold to 0.40 yields **99.52% recall** (only 219 missed attacks), readily outperforming Random Forest while maintaining superior operational efficiency.

---

### 4. Evaluation Metrics & Class Imbalance
#### Q13: Why is Accuracy an insufficient metric in Network Intrusion Detection?
**Answer:** Network traffic datasets are heavily class-imbalanced. A trivial model predicting the majority class can easily score 85%+ accuracy while missing 100% of stealthy attacks. Intrusion detection demands evaluation using **Recall** (minimizing breaches), **Precision** (minimizing alert fatigue), **$F_1$-Score**, and **Confusion Matrices**.

#### Q14: What is the difference between Precision and Recall in an IDS context?
**Answer:**
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — The percentage of actual attacks successfully caught. Low recall means missed intrusions (security breaches).
- **Precision**: $\frac{TP}{TP + FP}$ — The percentage of flagged alerts that were actually genuine attacks. Low precision triggers alert fatigue for SOC analysts.

#### Q15: Why is Attack Recall considered the most critical operational metric?
**Answer:** In cybersecurity, the operational cost of a **False Negative** (a hacker penetrating the network undetected) is catastrophic (data breach, ransomware, credential theft), whereas a **False Positive** merely requires analyst triage or automated secondary inspection.

#### Q16: What is the difference between Macro-$F_1$ and Weighted-$F_1$ in multiclass evaluation?
**Answer:**
- **Weighted-$F_1$** (0.7792 for XGBoost) computes the mean of class $F_1$ scores weighted by their sample support. Because dominant classes (Generic, Normal, Exploits make up ~81%) perform very well, weighted-$F_1$ remains high.
- **Macro-$F_1$** (0.5045 for XGBoost) computes the unweighted arithmetic mean across all 10 classes, treating rare classes (Worms: 44 test samples, Shellcode: 378 samples) with equal weight to majority classes (Normal: 37,000 samples). It explicitly exposes model performance on long-tailed minority threats.

#### Q17: How did the model perform on rare attacks like Worms and Shellcode?
**Answer:**
- **Worms (Extreme Rare, 44 test samples)**: XGBoost achieved **60.00% Precision** and **47.73% Recall** ($F_1 = 0.5316$), successfully identifying 21 out of 44 worm flows despite constituting only 0.05% of the data.
- **Shellcode (Rare, 378 test samples)**: XGBoost achieved **78.84% Recall** ($F_1 = 0.4585$), detecting 298 out of 378 shellcode attacks.
- Minority classes like `Analysis` and `Backdoor` remained the most difficult ($F_1 < 0.10$) because stealthy port probes and reverse shells mirror benign HTTP/SSH handshakes.

---

### 5. Threshold Analysis & Decision Trade-Offs
#### Q18: What is a Decision Threshold and why analyze it?
**Answer:** Standard binary classifiers output a probability $P(\text{Attack} \mid x) \in [0, 1]$ and apply a default cut-off threshold of $0.50$. In high-consequence domains like cybersecurity, analysts may adjust the threshold to tune the balance between sensitivity (catching stealthy threats) and specificity (reducing false alarms).

#### Q19: Why was 0.50 retained as the production default?
**Answer:**
- On a held-out validation split of training data, $F_1$-score peaked at threshold $0.50$ ($F_1 = 0.9669$).
- On the test set, $0.50$ achieved an optimal trade-off: **98.65% Attack Recall** while keeping false positives below 10k. Lowering to 0.30 raised recall to 99.87% but generated 14,786 false alarms; raising to 0.70 reduced false alarms but allowed nearly 2,000 undetected attacks.

---

### 6. Model Explainability (SHAP)
#### Q20: What is SHAP and how does TreeExplainer work?
**Answer:** SHAP (SHapley Additive exPlanations) is an axiomatic framework rooted in cooperative game theory. TreeExplainer computes exact Shapley values for tree ensembles in polynomial time by tracing all possible feature coalitions across tree decision paths, quantifying how each feature shifts the model's prediction away from the baseline expected value.

#### Q21: What are the top global features driving model predictions?
**Answer:**
1. `sttl` (Source Time-To-Live): Modulated TTL values indicate non-standard OS network stacks or port scanners.
2. `smean` (Mean Source Packet Size): Discriminates small probing payloads from legitimate data transfers.
3. `ct_srv_dst` & `ct_srv_src`: Connection density counts exposing reconnaissance sweeps and DoS flooding.
4. `synack`: Handshake latency spikes reflecting victim server buffer saturation.

#### Q22: What does a positive vs. negative SHAP value signify?
**Answer:**
- $\phi_i > 0$ indicates feature $i$ increased the log-odds output, **pushing the prediction toward Malicious Attack**.
- $\phi_i < 0$ indicates feature $i$ decreased the log-odds, **mitigating the score toward Normal traffic**.

#### Q23: Why is SHAP feature attribution NOT proof of causal relationship?
**Answer:** SHAP explains the mathematical dependency of the model's prediction on input features within the learned manifold. It does not establish physical real-world causation (e.g., high packet count might be correlated with an attack in the dataset without being the root cause of an exploit).

---

### 7. Real-World Limitations & Future Scope
#### Q24: What are the current limitations of this system?
**Answer:**
1. **Flow-level limitation**: Analyzes flow metadata rather than encrypted packet payloads.
2. **Concept drift**: Network traffic patterns evolve over time as new protocols and evasion techniques emerge.
3. **Imbalanced minority attacks**: Stealthy backdoor tunnels mimicking normal HTTPS remain difficult to separate without stateful application layer inspection.

#### Q25: What future improvements would you propose?
**Answer:**
1. Integration with streaming packet ingest (e.g., live PyShark or Zeek flow parsing).
2. Online/continual learning to adapt to zero-day concept drift.
3. Semi-supervised autoencoders for unsupervised novelty detection alongside supervised classification.
