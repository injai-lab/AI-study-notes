import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# 감정 라벨 정의

labels = {
    0: "기쁨",
    1: "슬픔",
    2: "분노",
    3: "불안",
    4: "놀람",
    5: "중립"
}

# 감정 분류기 모델 정의
class EmotionClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_labels=6):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.classifier(x)
    

# BERT 모델 & 토크나이저 로드(KLUE BERT)
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
bert = BertModel.from_pretrained("klue/bert-base")

# 감정 분류기 인스턴스 
classifier = EmotionClassifier()
classifier.eval() # 추론 모드

# 테스트용 문장
text = "너는 닿을수록 Thirsty 분명 가득한데 Thirsty Yeah, I got you boy \
Sip sip sipping all night 더 Deep deep deep in all night 얕은 수면보다 훨씬 \
짙은 너의 맘 끝까지 알고 싶어져 Sip sip sipping all night 더 Deep deep deep in all night\
맘이 커질수록 Thirsty"

# 토크나이즈 + 텐서 변환
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# BERT -> CLS 벡터 추출 -> 분류기 통과
with torch.no_grad():
    outputs = bert(**inputs)
    cls_vector = outputs.last_hidden_state[0][0]
    logits =classifier(cls_vector)
    predicted = torch.argmax(logits).item()

# 예측 결과 출력
print("문장 : ", text)
print("예측된 감정", labels[predicted])


torch.save(classifier.state_dict(), "emotion_classifier.pt")