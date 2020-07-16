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

#questions = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, ]*100)
#answers   = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, ]*100)

questAnsList = []

# 질문/정답 만들기 
questAnsList.append( QuestAns( [3.0, 2.0,  1.0], [1.0, -1.0] ) )
questAnsList.append( QuestAns( [1.0, 4.0, -3.0], [1.0, -1.0] ) )
questAnsList.append( QuestAns( [5.0, 5.0,  0.0], [1.0, -1.0] ) )
questAnsList.append( QuestAns( [8.0, 3.0,  5.0], [1.0, -1.0] ) )

questions = [ np.array( qa.quest, float ) for qa in questAnsList ]
answers = [ np.array( qa.answer, float ) for qa in questAnsList ]

questions = np.array( questions )
answers = np.array( answers )

print( "questions: ", questions )
print( "answers: ", answers ) 

#-- 학습 데이터 셋 만들기 

x0 = questions[0]
model = keras.models.Sequential( )
model.add( Dense(2, input_shape=x0.shape) )

model.compile( optimizer='adam', loss='mse', metrics=['accuracy'] )

epochs = 10_000
model.fit( questions, answers, epochs=epochs, batch_size=7 ) 

my_questions = questions[ 0:1, ] 

print( "\nMy Questions = " , my_questions )

my_answers = model.predict( my_questions )

print( "Answer = " , my_answers )

a = my_questions[ 0 ]
b = my_answers[ 0 ]
check = a[0]*b[0] + a[1]*b[1]
diff = a[2] - check

print( "check = ", check )
print( "diff = ", diff )

print( "Good bye!")