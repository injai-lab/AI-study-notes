# 감성 분석 RNN 프로젝트


이렇게 긍정이던 노래와, 부정적인 노래. 구분하는 감성 분석 RNN 프로젝트를 완료하였다.!

정말 신기한 것들이 많았고, 내가 좋아하는 노래들을 가지고 이 노래가 알고리즘이 정확하게 분석하였는지 예측 값과, 분류 결과를 통하여 눈 앞에 바로 보여지는 결과에 굉장히 흥미가 있었다.


🎧 감성 분석 RNN 프로젝트
노래 가사의 감정을 구분하는 감성 분석 RNN 프로젝트를 성공적으로 완료했다!

이번 프로젝트는 단순한 코드 구현이 아니라, 내가 좋아하는 노래 가사들을 데이터로 사용해 긍정적인 감정과 부정적인 감정을 분석하는 데 의미가 컸다.
특히, 모델이 문장을 읽고 감정을 예측하는 과정을 직접 눈으로 확인할 수 있었던 것이 무척 신기했고 흥미로웠다.

🔨 내가 직접 구현한 과정 요약
dataset.csv를 만들어 직접 가사 데이터를 정리

torchtext를 이용해 데이터 전처리 및 토큰화

RNN 모델 설계 후, 하이퍼파라미터 조절과 학습 진행

torch.save()로 모델 저장하고, 다시 불러와서 새로운 문장 예측

spacy를 이용해 실제 문장 입력 시 즉각적인 감정 분석 결과 출력


![image](https://github.com/user-attachments/assets/568393fc-1b88-465c-ba8d-703f680003b4)

📈 가장 인상 깊었던 점
예측 결과가 단순히 숫자로 끝나는 것이 아니라

"이 노래는 긍정적",

"이 가사는 부정"

이렇게 내가 자주 듣던 음악의 감정을 기계가 해석하는 모습은 정말 놀라웠다.



