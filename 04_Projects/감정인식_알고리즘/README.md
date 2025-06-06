# 📝 감정 인식 알고리즘

## 📝 감정 인식 알고리즘 실습 사유

이번 실습에서는 이전에 구현했던 RNN 기반 감정 분석 모델과는 다른 접근 방식을 통해, 

인공지능이 데이터를 학습하고 감정을 분류하는 과정을 이해하고자 한다. 

다양한 알고리즘을 적용하여 감정 인식이 어떻게 이루어지는지 비교·분석하는 데에 목적이 있다.


### 📦 사용한 패키지 (Requirements)
- Python 버전: `3.12.7`

VScode의 TERMINAL에서 

1. .\Venv\Scripts\activate 작성  --> 활성화 되어야 함.

---
- 주요 패키지:
  
    pip install transformers
  
    pip install datasets
  
    pip install konlpy




Anaconda Prompt에서 
"conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
작성.

VScode에서 

/*
import torch

print("CUDA 사용 가능 : ", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU 이름 : ", torch.cuda.get_device_name(0))
    
    print("CUDA 버전 : ", torch.version.cuda)
    
*/

작성하여 TERMINAL에서 잘 나오는지.

![image](https://github.com/user-attachments/assets/4cc3b0b8-5089-42af-9ee6-c09845156879)


------

## 📝 실습

![image](https://github.com/user-attachments/assets/e4128434-0dc3-4579-9a4b-b629c9a20562)


  1.klue/bert-base 모델 로드 성공 ✅

  2.입력 문장을 토크나이징 ✅

  3.BERT 모델 통과시켜 last_hidden_state 얻음 ✅

  4.그 중 [CLS] 토큰 (outputs[0][0]) 벡터 추출 ✅




## emotion_classifier.py

테스트해본 text 문장들

1. "오늘 너무 우울하고 기분이 안 좋아"
   // paul Anka - Diana
2. "사람들이 뭐라든 상관없어 / 나는 영원히 너와 함께하길 기도할 거야, 너와 나는 나무 위 새들처럼 자유로울 거야, 제발 내 곁에 있어 줘, 다이애나"
   // paul Anka - put your head on my shoulder
3. "너의 입술을 내 입술 가까이 해줘 한 번만 키스해 줄래, 자기야? 작별 인사의 키스 한 번이라도 그러다 우리, 사랑에 빠질지도 몰라"
   // aespa - thirsty
5. "너는 닿을수록 Thirsty 분명 가득한데 Thirsty Yeah, I got you boy Sip sip sipping all night 더 Deep deep deep in all night 얕은 수면보다 훨씬
    짙은 너의 맘 끝까지 알고 싶어져 Sip sip sipping all night 더 Deep deep deep in all night 맘이 커질수록 Thirsty"

![image](https://github.com/user-attachments/assets/1e61dcce-9fee-4071-9c43-2a312550a083)

결과
- 1. 슬픔
- 2. 불안
- 3. 불안
- 4. 기쁨

 | 항목               | 상태   |
| ---------------- | ---- |
| `KLUE-BERT` 불러오기 | ✅ 성공 |
| `[CLS]` 벡터 추출    | ✅ 성공 |
| 감정 분류기 연결        | ✅ 성공 |
| 문장 4개 예측 테스트     | ✅ 성공 |


## train_classifier.py


| 구성 요소     | 상태                     |
| --------- | ---------------------- |
| BERT 임베딩  | ✅ 작동                   |
| 분류기 정의    | ✅ 작동                   |
| GPU 전송    | ✅ 작동                   |
| 학습 루프     | ✅ 작동                   |
| 출력 정확도    | ✅ 출력됨                  |
| tokenizer | ✅ `AutoTokenizer`로 정상화 |

![image](https://github.com/user-attachments/assets/029098e4-75e6-4ad5-91fa-7fe8f39d7705)


     

