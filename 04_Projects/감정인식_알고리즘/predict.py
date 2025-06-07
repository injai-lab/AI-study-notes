import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

# 라벨 맵
label_map = {0 : "기쁨", 1 : "슬픔", 2 : "분노", 3 : "불안", 4: "놀람", 5 : "중립"}

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 및 BERT 불러오기
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
bert = BertModel.from_pretrained("klue/bert-base").to(device)

# 분류기 정의
class EmotionClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_labels=6):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.classifier(x)
    
# 모델 로딩
classifier = EmotionClassifier().to(device)
classifier.load_state_dict(torch.load("emotion_classifier.pt", map_location=device))
classifier.eval()

# 예측할 문장
text = input("문장을 입력하세요.")

# 토크나이즈
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 임베딩 -> 분류
with torch.no_grad():
    outputs = bert(**inputs)
    cls_vector = outputs.last_hidden_state[0][0]
    logits = classifier(cls_vector)
    predicted = torch.argmax(logits).item()

print("예측 감정 : ", label_map[predicted])