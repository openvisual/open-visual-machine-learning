# -*- coding: utf-8 -*-

import numpy as np
import random

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda  
from tensorflow.python.keras import backend as K 

model = keras.models.Sequential( )
model.add(Dense(1, input_shape=[1] )) 

class QuestAns :
    def __init__(self, quest, answer) :
        self.quest = quest
        self.answer = answer
    pass
pass

questAns = QuestAns( 2, 1 )

questAnsList = []

questAnsList.append( QuestAns( 2, 1 ) )
questAnsList.append( QuestAns( 3, 5 ) )
questAnsList.append( QuestAns( 4, 7 ) )

questions = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, ]*100)

answers   = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, ]*100)

model.compile( optimizer='adam', loss='mse', metrics=['accuracy'] )

epochs = 20 #30
model.fit( questions, answers, epochs=epochs, batch_size=7, )  

my_questions = [ 10, 0 , 20 ]

print( "\nMy Questions = " , my_questions )

my_answers = model.predict( my_questions )

print( "Answer = " , my_answers )