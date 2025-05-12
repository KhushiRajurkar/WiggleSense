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
  
## 📊 Plots
1) Wiggle Visualization (Sample Pose Image)
![image](https://github.com/user-attachments/assets/f53636d8-de64-426e-b625-57a34a31e89b)

2) Wiggle Loop Detection Line Plot
![image](https://github.com/user-attachments/assets/68707f9b-63de-405c-b0b4-fe47c90cd1bd)

3) Hive Health Score Bar Chart
![image](https://github.com/user-attachments/assets/e5fdd5fc-569d-4d08-a982-0ce12c7a392c)


## 🌼 Why This Matters
Bees are dying due to climate change and human activities. Understanding bee dances can help track pollination, detect environmental stress, and support conservation efforts. This project turns keypoint data into actionable behavior insights.
