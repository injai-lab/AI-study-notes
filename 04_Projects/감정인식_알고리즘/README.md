# 📝 감정 인식 알고리즘



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


klue/bert-base 모델 로드 성공 ✅

입력 문장을 토크나이징 ✅

BERT 모델 통과시켜 last_hidden_state 얻음 ✅

그 중 [CLS] 토큰 (outputs[0][0]) 벡터 추출 ✅
