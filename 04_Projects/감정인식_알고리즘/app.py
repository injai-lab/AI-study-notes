import streamlit as st
import torch
from transformers import BertTokenizer, BertModel

#감정 레이블 
label_map = {0: "기쁨", 1: "슬픔", 2: "분노", 3: "불안", 4: "놀람",5: "중립"}

# 모델 클래스
class EmotionClassifier(torch.nn.Module):
    def __init__(self, hidden_size=768, num_labels=6):
        super().__init__()
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.classifier(x)
    
# 모델 불러오기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
bert = BertModel.from_pretrained("klue/bert-base")
bert.to(device)


model = EmotionClassifier().to(device)
model.load_state_dict(torch.load("emotion_classifier.pt", map_location=device))
model.eval()

# Streamlit UI
st.title("감정 예측기")
text = st.text_area("문장을 입력하세요", "오늘 너무 행복해!")


if st.button("예측하기"):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)   
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert(**inputs)
        cls_vector = outputs.last_hidden_state[0][0]
        logits = model(cls_vector)
        pred = torch.argmax(logits).item()

    st.success(f"예측 감정 : **{label_map[pred]}**")