import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

TEXT = Field(tokenize = 'spacy', tokenizer_language='en_core_web_sm', lower=True)
LABEL = Field(sequential=False, use_vocab=False)

datafields = [('text', TEXT), ('label', LABEL)]

  # 1 차원배열 생성 (x(text), x(TEXT))

train_data, test_data = TabularDataset.splits(
      path='C:/Users/ansdy/Desktop/AI/04_Projects/RNN_AI_Learning/04_Projects/감성분석', train='dataset.csv', test='dataset.csv', format='csv',
      fields=datafields, skip_header=True)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size = 2,             #배치 크기(숫자를 바꾸면 4나 8로 크기변경 가능)
    sort_key= lambda x : len(x.text),
    sort_within_batch = False,
    device=torch.device('cpu')
    )    

TEXT.build_vocab(train_data)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
        
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))
    
    
    
for example in train_data :
    print(example.text)
    print(example.label)    


""" 하이퍼파라미터 설정 """
INPUT_DIM = len(TEXT.vocab)                  #입력층
EMBEDDING_DIM = 10
HIDDEN_DIM = 256                             #은닉층
OUTPUT_DIM = 1 # 이진 분류 0과 1              #출력층

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)  # 알고리즘 선택


""" 옵티마이저 & 손실함수 설정 """
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

""" 정확도 계산 함수 """
def binary_accuracy(preds, y) :
    rounded = torch.round(torch.sigmoid(preds))
    correct = (rounded == y).float()
    return correct.sum() / len(correct)

""" 학습 루프 (train function)"""
def train(model, iterator, optimizer, criteriion) :
    model.train()
    epoch_loss = 0
    epoch_acc = 0
        
    for batch in iterator :
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label.float())
        acc = binary_accuracy(predictions, batch.label.float())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

for epoch in range(5) :
    epoch_loss, epoch_acc = train(model, train_iterator, optimizer, criterion)
    print(f'epoch: {epoch+1} | loss: {epoch_loss:.3f} | acc: {epoch_acc:.2f}')


with open('TEXT_vocab.pkl', 'wb') as f:
    pickle.dump(TEXT, f)

""" 평가 함수(evaluate)"""
def evaluate(model, iterator, criterion)  :
    model.eval()    # 평가모드 Dropout, BatchNorm와 같은 것을 비활성화 시키는 평가 모드
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad(): # 평가 시에는 gradient 계산 X(불 필요한 메모리 사용을 줄여주는 것)
        for batch in iterator :
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label.float())
            acc = binary_accuracy(predictions, batch.label)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
            
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss : {test_loss:.3f} | Test Acc : {test_acc:.2f}')


torch.save(model.state_dict(), 'sentiment-model.pt')  // 모델 저장을 통해 불러오기 가능.
            




