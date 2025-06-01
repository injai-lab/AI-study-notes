## 📝 프로젝트 개발 메모

### 📦 사용한 패키지 (Requirements)
- Python 버전: `3.x`

-------

- 주요 패키지:
  - python --version
  - python -m pip install --upgrade pip
  - python -m pip install tensorflow matplotlib numpy

  --pip 업그레이드
  python -m ensurepip --upgrade
-


🧱 전체 구성 순서
데이터셋 로드 및 전처리 ✅

모델 생성 (Generator, Discriminator) 

손실 함수 정의 (binary crossentropy)

옵티마이저 설정

학습 루프 (train_step, train)

결과 이미지 저장 (generate_and_save_images)




----
### 📦 train.py(전체 구조 요약)
1. import 및 하이퍼파라미터 정의
2. 데이터 로드 및 전처리
3. 모델 생성 (Generator, Discriminator)
4. 손실 함수 정의
5. 옵티마이저 설정
6. 이미지 저장 함수 (generate_and_save_images)
7. train_step 함수
8. train 함수
9. train(train_dataset, EPOCHS) 실행
