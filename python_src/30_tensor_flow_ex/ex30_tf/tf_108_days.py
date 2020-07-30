# -*- coding: utf-8 -*-

# 0. 사용할 패키지 불러오기
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda  
from tensorflow.python.keras import backend as K
import random

# 1. 데이터셋 생성하기

x_train = [ 0, 1, 2, 3, 4, 5, 6 ]*1000
y_train = [ 1, 2, 3, 4, 5, 6, 0 ]*1000

x_test = [ 6, 0, 1, 2, 3, 4, 5 ]*100
y_test = [ 0, 1, 2, 3, 4, 5, 6 ]*100

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=1, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(1)) 

# 3. 모델 학습과정 설정하기
model.compile( optimizer='rmsprop', loss='mse' )

# 4. 모델 학습시키기
from tensorflow.keras.callbacks import EarlyStopping
# 랜덤 시드 고정 
np.random.seed(5)
early_stopping = EarlyStopping(patience = 20) # 조기종료 콜백함수 정의

hist = model.fit(x_train, y_train,
    epochs=50, batch_size=100, 
    callbacks=[early_stopping] 
    )

# 5. 학습과정 살펴보기
# %matplotlib inline
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.ylim(0.0, 1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 6. 모델 평가하기
loss = model.evaluate(x_test, y_test, batch_size=32)

print( '\nLoss : ' + str(loss))


# 7. 주어진 데이터로 추론 모드에서 마지막 층의 출력을 예측하여 넘파이 배열로 반환합니다:
print( "\n--- Result " )

my_quest =  [ 0, 1  ] 
my_answer = model.predict( my_quest )

print( "my question = ",  my_quest )
print( "my answer = ", my_answer )