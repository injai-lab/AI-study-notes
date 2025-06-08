import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 불러오기
df = pd.read_excel("korean_char_dataset.xlsx")

# 라벨 숫자로 변환
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Emotion"])

# 데이터 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Sentence"], df["label"], test_size=0.2, random_state=42
)

#데이터셋 클래스
class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels =zip(*batch)
    tokenized = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
    labels = torch.tensor(labels)
    return tokenized, labels

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
bert = BertModel.from_pretrained("klue/bert-base").to(device)

# 분류기 정의
class EmotionClassifierWithMeanPooling(nn.Module):
    def __init__(self, hidden_size=768, num_labels=7):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_state):
        x = hidden_state.mean(dim=1)
        x = self.dropout(x)
        x = self.relu(x)
        return self.classifier(x)        

classifier = EmotionClassifierWithMeanPooling().to(device)
optimizer = optim.Adam(list(classifier.parameters()) + list(bert.parameters()), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# DataLoader

batch_size = 8
train_dataset = EmotionDataset(train_texts, train_labels)
val_dataset = EmotionDataset(val_texts, val_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# 학습 루프
epochs = 10
for epoch in range(epochs):
    classifier.train()
    bert.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        outputs = bert(**inputs)
        logits = classifier(outputs.last_hidden_state)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim = 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    print(f"[Epoch {epoch+1}] Loss : {total_loss:.4f} | Accuracy : {acc:.4f}")


# 검증
classifier.eval()
bert.eval()
val_correct = 0
val_total = 0
with torch.no_grad():
    for inputs, lables in val_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels =  lables.to(device)
        outputs = bert(**inputs)
        logits = classifier(outputs.last_hidden_state)
        preds = torch.argmax(logits, dim=1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)
val_acc = val_correct / val_total

print(f"[Validation] Accuracy : {val_acc : .4f}")
# labels(Emotion)이 몇 개로 인식하는지 나오는 문장.
print(label_encoder.classes_)
print(df["label"].unique())

# 모델 저장
torch.save(classifier.state_dict(), "emotion_classifier.pt")