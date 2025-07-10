# UAV Digital Twin: Trustworthy Evolution Evaluation & State Prediction

This repository is dedicated to **trustworthy evolution evaluation** in digital twin systems, with a core focus on **Unmanned Aerial Vehicle (UAV) state prediction**. It aims to provide reliable model evolution analysis and accurate UAV state forecasting for digital twin applications.


## 🌟 Key Objectives
- Implement trustworthy evolution evaluation mechanisms for UAV digital twins.
- Achieve precise multi-step state prediction of UAVs to support digital twin dynamics.


## 📊 Dataset
The project leverages the [Pelican Dataset](https://github.com/wavelab/pelican_dataset) from WAVELab, which contains rich flight data (e.g., position, orientation, motor commands) of quadrotor UAVs, providing a solid foundation for model training and validation.


## 🚀 Methodology
Our approach combines offline learning and online adaptation to ensure robust performance:
- **Offline Pre-training**: The prediction model is initially trained on the Pelican Dataset to learn fundamental UAV dynamics and pose characteristics, enabling baseline multi-step pose prediction.
- **Online Learning Evolution**: To adapt to real-world changes and maintain long-term accuracy, the model is continuously updated using an online learning-based evolution method, ensuring the digital twin evolves in line with the physical UAV.


## 📌 Applications
This work supports UAV digital twin research in scenarios such as:
- Real-time state monitoring
- Predictive maintenance
- Dynamic performance evaluation
- Evolutionary model validation
