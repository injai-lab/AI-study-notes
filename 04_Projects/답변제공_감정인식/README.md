# 📝 감정 인식 알고리즘


## 📝 감정 인식 알고리즘 실습 사유

이번 실습에서는 이전에 구현했던 RNN 기반 감정 분석 모델과는 다른 접근 방식을 통해,

인공지능이 데이터를 학습하고 감정을 분류하는 과정을 이해하고자 한다.

다양한 알고리즘을 적용하여 감정 인식이 어떻게 이루어지는지 비교·분석하는 데에 목적이 있다.

---



### 📦 사용한 패키지 (Requirements)
- Python 버전: 3.12.7

VScode의 TERMINAL에서

.\Venv\Scripts\activate 작성 --> 활성화 되어야 함.
주요 패키지:

pip install transformers

pip install datasets

pip install konlpy

## 📝 train_classifier.py (emotion_classifier.pt)

![image](https://github.com/user-attachments/assets/c7a5566c-4176-4f47-9991-373ae23dedbc)

그래픽카드가 갈려나갔다... ㅋㅋㅋ
(RTX 2060) epochs = 10을 돌려서

최종  =  loss : 340.0656, accuracy : 0.9701(97%)

torch.save(classifier.state_dict(), "emtion_classifier.pt")
작성하여 모델로 저장.




