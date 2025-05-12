# WiggleSense: Bee Behavior Analysis Using Pose Data & ML

**WiggleSense** is a machine learning pipeline that decodes bee waggle dances from pose estimation keypoints. It classifies wiggle behavior, predicts direction of foraging, and estimates hive health using statistical and visual methods.

## Contents
- `Wiggle_Sense.ipynb` — main pipeline (pose extraction → angle calc → ML → health scoring)
- `final_bee_wiggle_features.csv` — processed pose + movement data
- `final_bee_health_analysis.csv` — includes smoothed angles, variance, wiggle tags, and hive health scores
- `requirements.txt` — dependencies

## Techniques Used

1) **Pose Vector Extraction** (`dx`, `dy`)  
  Computed directional vectors between bee head and tail keypoints using 2D Euclidean geometry, enabling motion orientation analysis from pose frames.

2) **Angle Computation via `atan2()`**  
  Translated spatial vectors into absolute body orientation angles (in degrees), preserving both magnitude and directional sign for accurate behavioral mapping.

3) **Signal Smoothing & Rolling Variance Analysis**  
  Applied rolling window smoothing (mean) over body angle sequences to denoise movement trends, followed by variance computation to detect rhythmic oscillations (i.e. waggles).

4) **Wiggle Loop Detection (Variance-Based Classification)**  
  Identified high-frequency motion zones using a variance threshold (~20–30°), classifying segments with consistent angular fluctuation as **wiggle dances**, crucial for foraging behavior.

5) **Random Forest Regression**  
  Trained a regression model to predict the bee's orientation (`angle_deg`) from head-tail position features (`dx`, `dy`), showcasing direction estimation from pose data.

6) **Wiggle Classification with Random Forest Classifier**  
  Supervised model trained to distinguish wiggle frames vs. non-wiggle based on statistical movement features — ideal for expanding into sequence-level behavior classification.

7) **Data-Driven Health Scoring using Quantiles**  
  Adaptively labeled each bee’s activity as **Healthy**, **Low Activity**, or **Stressed** based on percentile thresholds of angular variance across the dataset, ensuring thresholds are specific to data distribution.

8) **Behavioral Visualization**  
  Generated frame-wise line plots and stacked bar charts to track individual bee behavior (body angle, wiggle pattern) and population-level hive health.


  
## Plots
1) Wiggle Visualization (Sample Pose Image)
![image](https://github.com/user-attachments/assets/f53636d8-de64-426e-b625-57a34a31e89b)

2) Wiggle Loop Detection Line Plot
![image](https://github.com/user-attachments/assets/68707f9b-63de-405c-b0b4-fe47c90cd1bd)

3) Hive Health Score Bar Chart
![image](https://github.com/user-attachments/assets/e5fdd5fc-569d-4d08-a982-0ce12c7a392c)


## 🌼 Why This Matters
Bees are dying due to climate change and human activities. Understanding bee dances can help track pollination, detect environmental stress, and support conservation efforts. This project turns keypoint data into actionable behavior insights.
