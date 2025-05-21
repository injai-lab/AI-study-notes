import torch
import torch.nn as nn
import spacy
import pickle


# RNN 클래스 재정의
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

# 하이퍼파라미터 (학습 당시와 동일해야 함)
INPUT_DIM = 71
EMBEDDING_DIM = 10
HIDDEN_DIM = 256
OUTPUT_DIM = 1

# 모델 불러오기
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load('sentiment-model.pt'))
model.to(torch.device('cpu'))
model.eval()

with open('TEXT_vocab.pkl', 'rb') as f:
    TEXT = pickle.load(f)

nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence, TEXT, device) :
    model.eval()
    tokens = [tok.text.lower() for tok in nlp(sentence)] #소문자 토크나이징
    indices = [TEXT.vocab.stoi[token] for token in tokens] # 단어 인덱스로 변환
    tensor = torch.LongTensor(indices).unsqueeze(1).to(device) #[문장길이, 1]로 변형
    prediction = torch.sigmoid(model(tensor)) #시그모이드로 확률 반환
    return prediction.item() #float값 리턴.


# sentence = "Oh-oh, ooh You've been runnin' 'round, runnin' 'round, runnin' 'round throwin' that dirt all on my name 'Cause you knew that I, knew that I, knew that I'd call you up You've been going 'round, going 'round, going 'round every party in L.A. 'Cause you knew that I, knew that I, knew that I'd be at one, oh I know that dress is karma, perfume regret You got me thinking 'bout when you were mine, oh And now I'm all up on ya, what you expect? But you're not coming home with me tonight You just want attention, you don't want my heart Maybe you just hate the thought of me with someone new Yeah, you just want attention, I knew from the start You're just making sure I'm never gettin' over you"
# sentence = "I'm so young and you're so old This, my darling I've been told I don't care just what they say 'Cause forever I will pray You and I will be as free As the birds up in the trees Oh,  lease, stay by me, Diana  Thrills I get when you hold me close Oh, my darling, you're the most I love you but do you love me"
# sentence = "Cause I'm off my face, in love with you I'm out my head, so into you And I don't know how you do it But I'm forever ruined by you"
sentence = "Fly me to the moon Let me play among the stars Let me see what spring is like on Jupiter and Mars"
# sentence = "I see the crystal raindrops fall And the beauty of it all is when the sun comes shining through to make those rainbows in my mind"
# sentence = "There is the sun and moon They sing their own sweet tune Watch them when dawn is due Sharing one space Some walk by night Some fly by day Some think it's sweeter When you meet along the way"
# sentence = "너는 닿을수록 Thirsty 분명 가득한데 Thirsty Yeah, I got you boy Sip sip sipping all night 더  Deep deep deep in all night 얕은 수면보다 훨씬 짙은 너의 맘 끝까지 알고 싶어져Sip sip sipping all night 더 Deep deep deep in all night 맘이 커질수록 Thirsty"
# sentence = "어느새 빗물이 내 발목에 고이고 참았던 눈물이 내 눈가에 고이고 I cry그대는 내 머리 위의 우산 어깨 위에 차가운 비 내리는 밤 내 곁에 그대가 습관이 돼버린 나 난 그대 없이는 안 돼요."
# sentence = "그대 먼저 헤어지자 말해요 나는 사실 그대에게 좋은 사람이 아녜요 그대 이제 날 떠난다 말해요 잠시라도 이 행복을 느껴서 고마웠다고"
# sentence = "Yesterday all my troubles seemed so far away Now it looks as though they're here to  tay Oh, I believe in yesterday Suddenly I'm not half the man I used to be There's a shadow hanging over me Oh, yesterday came suddenly Why she had to go I don't know, she wouldn't say I said something wrong now I long for yesterday Yesterday love was such an easy game to play Now I need a place to hide away Oh, I believe in yesterday. Why she had to go I don't know, She wouldn't say I said something wrong Now I long for yesterday Yesterday love was such an easy game to play Now I need a place to hide away Oh, I believe in yesterday"
# sentence = "Don't ever leave me Don't say goodbye If you do, I'll be the lonely one I'll sit down and  will cry Don't ever tell me Don't say we're through If you do, I'll be the lonely one I'll sit down' and I'll cry over you No matter what you should say to me No matter what you do It would mean nothing at all to me Because I'm so in love with you (I love you)Don't ever change dear Oh Don't say we're through If you do, I'll be the lonely one And I'll sit down and I'll cry over you Oh I'll sit down and I'll cry over you"


pred = predict_sentiment(model, sentence, TEXT, torch.device('cpu'))
print(f"입력문장: {sentence}")
print(f"예측 값: {pred:.4f}")
print(f"분류 결과 : {'긍정' if pred >= 0.5 else '부정'}")