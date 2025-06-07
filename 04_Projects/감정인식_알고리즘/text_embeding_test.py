from transformers import BertTokenizer
from transformers import BertModel
import torch

# KoBERT 토크나이저 및 모델 불러오기

tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
model = BertModel.from_pretrained("klue/bert-base")

# 테스트 문장
text = "오늘 기분이 정말 좋지 않아..."

# 토큰화 미치 텐서 변환
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 모델 통과시켜 임베딩 열기
with torch.no_grad():
    outputs = model(**inputs)

# 출력 형태 확인

print("토큰 IDs : ", inputs['input_ids'])
print("출력 임베딩 shape : ", outputs.last_hidden_state.shape)
print("CLS 토큰 벡터", outputs.last_hidden_state[0][0])