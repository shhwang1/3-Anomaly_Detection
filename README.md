# Anomaly Detection Tutorial

## Table of Contents

#### 0. Overview of Anomaly Detection
___
### Density-based Anomaly Detection
#### 1. Local Outlier Factor (LOF) 
***
### Distance-based Anomaly Detection
#### 2. K-Neareast Neighbor Anomaly Detection
*****
### Model-based Anomaly Detection
#### 3. Auto-encoder
#### 4. Isolation Forest (IF)
___

## 0. Overview of Anomaly Detection
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/201886259-e8dafab7-55fe-480a-8428-e131f93ee1cc.png" width="600" height="400"></p>   

### - What is "Anomaly?"
Let's look at the beer bottles above. Beer bottles seem to be neatly placed, but you can see an unusual bottle of beer. Like this picture,  small number of unusual patterns of data present in this uniform pattern are called 'Anomaly'. This discovery of Anomaly is called Anomaly Detection, and there are some very important moments to utilize this task. Think about the manufacturing process of making cheap screws. The 'Anomaly' screws that occur during this process do not appear to be very fatal because the value per piece is very small. However, 'Anomaly' semiconductors that occur in expensive semiconductor processes are very expensive per unit, so even a small number of Anomaly products are very damaging to the process. That's why it's important to find the anomaly well.

### - What is difference between "Classification" & "Anomaly Detection"?
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/201891495-1dc08074-9e6d-4132-90fe-cf012bc63c39.png" width="600" height="300"></p> 

If so, you may be curious about the difference between Classification and Anomaly Detection. Classification and Anomaly Detection problems differ by the assumptions below.

### 1. Anomaly Detection utilizes very unbalanced data.
In explaining the definition of Anomaly, he mentioned the purpose of Anomaly Detection. Anomaly Detection aims to detect a small number of anomaly data embedded between the majority of data that are even. Therefore, we also use dataset consisting of very few ano data and most normal data.
