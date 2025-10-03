import os, zipfile, json, textwrap, shutil

# Base directory for Day 8 project
base_dir = "/mnt/data/Cloud-Log-Anomaly-Detection-Deepak-Kumar"
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "scripts"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "notebooks"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)

# Sample dataset (cloud_logs.csv)
logs_csv = """timestamp,ip,request_count,error_rate
2025-10-01 10:00:00,192.168.0.1,120,0.01
2025-10-01 10:01:00,192.168.0.2,130,0.02
2025-10-01 10:02:00,192.168.0.3,5000,0.50
2025-10-01 10:03:00,192.168.0.4,150,0.01
2025-10-01 10:04:00,192.168.0.5,140,0.02
2025-10-01 10:05:00,192.168.0.6,6000,0.70
"""
with open(os.path.join(base_dir, "data", "cloud_logs.csv"), "w") as f:
    f.write(logs_csv)

# Script: anomaly_detection.py
script = textwrap.dedent("""\
    \"\"\"Cloud Log Anomaly Detection - Deepak Kumar\"\"\"

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    import os

    os.makedirs("images", exist_ok=True)

    # Load dataset
    df = pd.read_csv("data/cloud_logs.csv", parse_dates=["timestamp"])

    # Features
    X = df[["request_count", "error_rate"]]

    # Train Isolation Forest
    model = IsolationForest(contamination=0.2, random_state=42)
    df["anomaly"] = model.fit_predict(X)

    # Map anomaly labels (-1 = anomaly, 1 = normal)
    df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

    # Save results
    df.to_csv("data/logs_with_anomalies.csv", index=False)

    # Plot anomalies
    plt.figure(figsize=(8,5))
    colors = df["anomaly"].map({"Normal":"blue", "Anomaly":"red"})
    plt.scatter(df["request_count"], df["error_rate"], c=colors)
    plt.xlabel("Request Count")
    plt.ylabel("Error Rate")
    plt.title("Cloud Log Anomalies - Deepak Kumar")
    plt.text(0.5, -0.15, 'Deepak Kumar', fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig("images/anomalies.png")
    plt.close()

    print("Anomaly detection complete! Check images/ and data/ folder.")
    """)
with open(os.path.join(base_dir, "scripts", "anomaly_detection.py"), "w") as f:
    f.write(script)

# Notebook: anomaly_detection.ipynb
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Cloud Log Anomaly Detection - Deepak Kumar\n\nDetect anomalies in synthetic cloud logs using Isolation Forest."]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["import pandas as pd\n\nlogs = pd.read_csv('data/cloud_logs.csv')\nlogs.head()"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
with open(os.path.join(base_dir, "notebooks", "anomaly_detection.ipynb"), "w") as f:
    json.dump(notebook, f, indent=2)

# README.md
readme = textwrap.dedent("""\
    # Cloud Log Anomaly Detection - Deepak Kumar

    This project detects anomalies in synthetic cloud server logs using Isolation Forest.

    ## Features
    - Synthetic cloud log dataset (timestamp, ip, request_count, error_rate)
    - Isolation Forest anomaly detection
    - Anomaly vs Normal logs visualization
    - Predictions saved to CSV
    - Watermark "Deepak Kumar" on plots

    ## Tech Stack
    - Python
    - Pandas, Numpy
    - Scikit-learn
    - Matplotlib

    ## How to Run
    ```bash
    pip install -r requirements.txt
    python scripts/anomaly_detection.py
    ```

    ## Output
    - Anomalies plot (images/anomalies.png)
    - Logs with anomaly labels (data/logs_with_anomalies.csv)

    ## Author
    Deepak Kumar
    """)
with open(os.path.join(base_dir, "README.md"), "w") as f:
    f.write(readme)

# requirements.txt
reqs = "pandas\nnumpy\nscikit-learn\nmatplotlib\n"
with open(os.path.join(base_dir, "requirements.txt"), "w") as f:
    f.write(reqs)

# Zip the repo
zip_path = "/mnt/data/Cloud-Log-Anomaly-Detection-Deepak-Kumar.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, base_dir)
            zf.write(file_path, arcname)

zip_path
