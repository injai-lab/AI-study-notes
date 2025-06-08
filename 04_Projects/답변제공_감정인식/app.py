import streamlit as st
import torch
from transformers import BertTokenizer, BertModel

# 레이블 매핑
label_map = {
    0: "공포", 1: "놀람", 2: "분노", 3: "슬픔", 
    4: "중립", 5: "행복", 6: "혐오"
}

# 모델 클래스 정의
class EmotionClassifier(torch.nn.Module):
    def __init__(self, hidden_size=768, num_labels=6):
        super().__init__()
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        
    def forward(self, x):
        return self.classifier(x)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
bert = BertModel.from_pretrained("klue/bert-base").to(device)

model = EmotionClassifier()
model.load_state_dict(torch.load("emotion_classifier.pt", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.title("감정 예측 챗봇")
text = st.text_area("문장을 입력하세요", "오늘 너무 행복해!")

if st.button("예측하기"):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert(**inputs)
        cls_vector = outputs.last_hidden_state[0][0]
        logits = model(cls_vector)
        pred = torch.argmax(logits).item()
    st.success(f"예측된 감정: **{label_map[pred]}**")


