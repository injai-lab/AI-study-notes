import matplotlib.pyplot as plt
import numpy as pd
import tensorflow as tf

# 1. 데이터셋 로드 및 기본 정보 출력
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print("Train images shape:", train_images.shape)  # (60000, 28, 28)
print("Test images shape:", test_images.shape)    # (10000, 28, 28)

# 2. 첫 번째 이미지 확인 (이미지 창에 출력됨)
plt.imshow(train_images[0], cmap="Greys")
plt.title("첫 번째 훈련 이미지")
plt.show()

# 3. 데이터 전처리: 이미지 reshape 및 정규화, 레이블 원-핫 인코딩
train_images = train_images.reshape((60000, 784))
train_images = train_images.astype('float32') / 255.0

test_images = test_images.reshape((10000, 784))
test_images = test_images.astype('float32') / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 4. 모델 구성: 입력층 → 은닉층(relu) → 출력층(softmax)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

# 5. 모델 컴파일: 옵티마이저, 손실 함수, 평가 지표 설정
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

# 6. 모델 학습: history에 학습 기록 저장 (검증 데이터도 같이 지정)
history = model.fit(train_images, train_labels,
                    epochs=5,
                    batch_size=128,
                    validation_data=(test_images, test_labels))

# 7. 모델 평가: 테스트 데이터셋으로 평가 후 정확도 출력
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("테스트 정확도:", test_acc)

# 8. 학습 결과 시각화: 손실과 정확도 그래프 출력
loss = history.history['loss']
acc = history.history['accuracy']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.title('학습 손실 및 정확도')
plt.legend()
plt.show()

import cv2 as cv


image = cv.imread('C:\\Users\\ansdy\\Desktop\\images\\Test_6.png', cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (28, 28))
image = image.astype('float32')
image = image.reshape(1, 784)
image = 255-image
image /= 255.0

plt.imshow(image.reshape(28, 28), cmap='Greys')
plt.show()

pred = model.predict(image, batch_size=1)
print("추정된 숫자 = ", pred.argmax())
print("2106016_황인재")


