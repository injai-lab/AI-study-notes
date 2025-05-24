🚢🚢🚢🚢🚢🚢🚢🚢🚢
# 타이타닉 생존 예측 프로젝트

 
📌 프로젝트 개요
본 프로젝트는 Titanic 탑승자 데이터를 기반으로 생존 여부를 예측하는 머신러닝 모델을 구축하는 데 목적이 있습니다. 데이터는 Kaggle의 Titanic: Machine Learning from Disaster 대회를 기반으로 하였습니다.

-----------------------------------------------------------------------------------------------------
💡 모델링 과정 요약

      데이터 전처리

      결측치 처리 (Age, Embarked, Fare 등)

      범주형 변수 인코딩 (Sex, Embarked)

      특성 선택 및 스케일링

      모델 선택 및 학습

      Logistic Regression

      RandomForestClassifier

      K-Nearest Neighbors (선택)
-----------------------------------------------------------------------------------------------------
💡모델 평가

Accuracy, Confusion Matrix, Cross-validation


   - Accuracy    정확도.
   - Confusion Matrix  2차원 배열
   - Cross-validation  교차 검증

-----------------------------------------------------------------------------------------------------

테스트셋 예측 및 제출 파일 생성

submission.csv

-----------------------------------------------------------------------------------------------------

📈 주요 변수 설명

변수명	           설명
----------------------------
Pclass	      객실 등급 (1~3)
Sex	              성별
Age	              나이
SibSp	        형제/배우자 수
Parch	         부모/자녀 수
Fare	            운임
Embarked	   승선 항구 (C/Q/S)

-----------------------------------------------------------------------------------------------------
titanic.py
![image](https://github.com/user-attachments/assets/69ca9d71-0d20-4f5b-99e3-83ff05cada71)




모델의 정확도(accuracy)가
"Accuracy :  0.8212290502793296"
높게나왔다.
-----------------------------------------------------------------------------------------------------


Titanic_Project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv (baseline)
├── titanic.py
├── README.md
└── requirements.txt


