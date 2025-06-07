import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import AutoTokenizer
from transformers import BertModel
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("dataset_1000.csv")
texts = df["text"].tolist()
# 감정 라벨 매빙
label_map = {"기쁨" : 0, "슬픔" : 1, "분노" : 2, "불안" : 3, "놀람" : 4, "중립" : 5}
labels =[label_map[label] for label in df["label"]]

# 샘플 데이터 셋
texts = [
    "오늘은 기분이 너무 좋아!", 
    "정말 짜증나는 하루였어.",
    "너무 우울하고 눈물이 나.",
    "긴장돼서 아무것도 못 하겠어",
    "헉 진짜 놀랐어 이게 실화야?",
    "음 그냥 평범한 하루인 듯?"
]

# labels = ["기쁨", "분노", "슬픔", "불안", "놀람", "중립"]
# labels = [label_map[label] for label in labels]

# BERT 구성
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
bert = BertModel.from_pretrained("klue/bert-base").to(device)

# 분류기 정의
class EmotionClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_labels=6):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, x) :
        return self.classifier(x)
    
classifier = EmotionClassifier().to(device)

#옵티 마이저 & 손실함수
optimizer = optim.Adam(classifier.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

#학습 루프
epochs = 10
for epochl in range(epochs):
    total_loss = 0
    correct = 0

    for text, label in zip(texts, labels):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert(**inputs)
        cls_vector = outputs.last_hidden_state[0][0]

        logits = classifier(cls_vector)
        loss = criterion(logits.unsqueeze(0), torch.tensor([label]).to(device))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(logits).item()
        if predicted == label:
            correct += 1
for epoch in range(epochs) :
    print(f"[Epoch {epoch+1}] Loss : {total_loss:.4f} | Accuracy : {correct}/{len(texts)}")


torch.save(classifier.state_dict(), "emotion_classifier.pt")
print("모델 저장 완료")


