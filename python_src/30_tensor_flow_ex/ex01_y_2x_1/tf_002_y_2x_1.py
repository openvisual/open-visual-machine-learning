# -*- coding: utf-8 -*-
import warnings 
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np, random

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda  
from tensorflow.python.keras import backend as K 

class QuestAns :
    def __init__(self, quest, answer) :
        self.quest = quest
        self.answer = answer
    pass
pass

# 학습 데이터셋 만들기 

a = [ 'a', 'b' ]*100
print( a )
b = "ab"*100
print( b )

#questions = np.array( [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, ]*10 )
#answers   = np.array( [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, ]*10 )

# [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, ]*3

# [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, ] 

a =  np.array( [1,2,3,4] )
print( a )

for i, v in enumerate( a ):
    a[i] = v*123
pass

a = a*100

print( a )

questAnsList = []

# 질문/정답 만들기 
questAnsList.append( QuestAns( -1.0, -3.0 ) )
questAnsList.append( QuestAns( 0.0, -1.0 ) )
questAnsList.append( QuestAns( 1.0, 1.0 ) )
questAnsList.append( QuestAns( 2.0, 1.0 ) )
questAnsList.append( QuestAns( 3.0, 5.0 ) )
questAnsList.append( QuestAns( 4.0, 7.0 ) )
questAnsList.append( QuestAns( 5.0, 9.0 ) )

questions = [ qa.quest for qa in questAnsList ]
answers = [ qa.answer for qa in questAnsList ]

print( "questions: ", questions )
print( "answers: ", answers )

questions = questions*100
answers = answers*100

#-- 학습 데이터 셋 만들기 

# y = 2*x - 1
# x = 질문, y  = 정답 
model = keras.models.Sequential( )
model.add(Dense(1, input_shape=[1] )) 

#loss = "mean_absolute_error"
#loss = "mse"
loss = "mae"

model.compile( optimizer='adam', loss=loss, metrics=['accuracy'] )

epochs = 30
model.fit( questions, answers, epochs=epochs, batch_size=7 )  

my_questions = [ 10, 0 , 20 ]

print( "\nMy Questions = " , my_questions )

my_answers = model.predict( my_questions )

print( "Answer = " , my_answers )