# Cyber Anomaly Detection

An end-to-end anomaly detection pipeline and SIEM-style dashboard, demonstrating data engineering, machine learning, and cybersecurity analytics.

## Features

- **Data Ingestion & Cleaning** via Python scripts  
- **Anomaly Detection Model** built with pandas and scikit-learn  
- **Threshold Calibration** to select alert cut-offs  
- **Interactive Analyst Dashboard** in Streamlit (time-series, top hosts, score distribution)  
- **SIEM Integration** using Elasticsearch & Kibana dashboards  
- **Automated Alerting** via Kibana Detection Rules  
- **Containerized** with Docker (Streamlit app + ELK stack)

## Repository Structure

```text
cyber-anomaly-detection/
├─ dashboard/               ← Streamlit app  
│  └─ 04_dashboard.py  
├─ scripts/                 ← ETL, modeling & calibration  
│  ├─ 01_data_loading.py  
│  ├─ 02_anomaly_detection_modeling.py  
│  └─ 03_threshold_calibration.py  
├─ data/  
│  ├─ raw/                  ← source CSVs  
│  └─ processed/            ← anomaly_scores.csv  
├─ reports/                 ← project summary, slides, screenshots  
│  ├─ .gitkeep  
│  └─ project_summary.md  
├─ docker-compose.yml       ← Elasticsearch & Kibana services  
├─ Dockerfile               ← containerizes Streamlit app  
├─ requirements.txt         ← Python deps for Streamlit & scripts  
└─ README.md                ← this guide
