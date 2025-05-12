# 🐝 WiggleSense: Bee Behavior Analysis Using Pose Data & ML

**WiggleSense** is a machine learning pipeline that decodes bee waggle dances from pose estimation keypoints. It classifies wiggle behavior, predicts direction of foraging, and estimates hive health using statistical and visual methods.

## 📂 Contents
- `Wiggle_Sense.ipynb` — main pipeline (pose extraction → angle calc → ML → health scoring)
- `final_bee_wiggle_features.csv` — processed pose + movement data
- `final_bee_health_analysis.csv` — includes smoothed angles, variance, wiggle tags, and hive health scores
- `requirements.txt` — dependencies

## 🧠 Techniques Used
- Pose vector extraction (dx, dy)
- Angle computation via `atan2()`
- Random Forest regression (direction)
- Variance-based wiggle loop detection
- Data-driven health scoring using quantiles

## 🌍 Why This Matters
Bees are dying due to climate change and human activities. Understanding bee dances can help track pollination, detect environmental stress, and support conservation efforts. This project turns keypoint data into actionable behavior insights.
