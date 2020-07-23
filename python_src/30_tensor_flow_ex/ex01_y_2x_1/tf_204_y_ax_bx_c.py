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

questAnsList = []

# 질문/정답 만들기 
questAnsList.append( QuestAns( [3.0, 2.0,  1.0], [1.0, -1.0] ) )
questAnsList.append( QuestAns( [1.0, 4.0, -3.0], [1.0, -1.0] ) )
questAnsList.append( QuestAns( [5.0, 5.0,  0.0], [1.0, -1.0] ) )
questAnsList.append( QuestAns( [8.0, 3.0,  5.0], [1.0, -1.0] ) )

questions = [ qa.quest for qa in questAnsList ]
answers = [ qa.answer for qa in questAnsList ]

questions = np.array( questions )
answers = np.array( answers )

print( "questions: ", questions )
print( "answers: ", answers ) 

#-- 학습 데이터 셋 만들기 

model = keras.models.Sequential()

input_shape = questions[0].shape
output_count = answers[0].shape[0]

print( "input shape  = %s" % input_shape )
print( "output count = %s" % output_count )

# 신경망 만들기 
model.add( Dense(output_count, input_shape=input_shape) )

model.compile( optimizer='adam', loss='mae', metrics=['accuracy'] )

epochs = 8_000
model.fit( questions, answers, epochs=epochs, batch_size=7 ) 

my_questions = np.array( [ [ 3., 2., 1. ] , ] )

print( "\nMy Questions = " , my_questions )

my_answers = model.predict( my_questions )

print( "Answer = " , my_answers )

a = my_questions[ 0 ]
b = my_answers[ 0 ]
check = a[0]*b[0] + a[1]*b[1]
diff = a[2] - check

print( "check = %.4f" % check )
print( "diff  = %.4f" % diff  )

print( "\nGood bye!")