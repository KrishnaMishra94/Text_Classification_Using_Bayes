import pandas as pd
import numpy as np


################ FUNCTIONS #######################

def draw_missing_values_table(df):
    nullCount  = df.isnull().sum().sort_values(ascending=False)
    percentage = (df.isnull().sum().sort_values(ascending=False))*100/df.shape[0]
    missingTable = pd.concat([nullCount,percentage],axis=1,keys=['Total','Percentage'])
    return missingTable


def preprocess_text(df,column):
    import re
    for i in range(len(df)):
        ######  REMOVING SPECIAL CHARACTERS
        df.loc[i,column]  = re.sub(r'\W',' ',str(df.loc[i,column]))
    
        ######  REMOVING ALL SINGLE CHARACTERS
        df.loc[i,column]  = re.sub(r'\s+[a-zA-Z]\s+',' ',str(df.loc[i,column]))
    
        ######  REMOVING MULTIPLE SPACES WITH SINGLE SPACE
        df.loc[i,column]  = re.sub(r'\s+',' ',str(df.loc[i,column]))
        
    return df


###################################################

################## READING INPUT FILE ######################

data = pd.read_excel('DA_Test.xlsx')

############################################################



################# CHECK FOR MISSING VALUES FOR INPUT & OUTPUT COLUMN ###########

input_column  = 'Particulars'
output_column = 'First Level Classification'
draw_missing_values_table(data.loc[:,[input_column,output_column]])

##############################################################################


################ PERFORMING PREPROCESSING OF INPUT TEXT ###################
data[input_column].head()

data = preprocess_text(data,input_column)

data[input_column].head()
##########################################################################


################# DIVIDING DATA INTO INPUT OUTPUT ######################

X = data.loc[:,input_column]
y = data.loc[:,output_column]
#######################################################################


############### USING BAG OF WORDS MODEL TO CONVERT FEATURES INTO NUMBERS ############
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_counts   = count_vect.fit_transform(X).toarray()

######################################################################################


############### TRAIN - TEST DATA SPLIT #################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.3, random_state=0)

########################################################



################ PERFORMING NAIVE BAYESIAN CLASSIFICATION ##############

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)

predict = clf.predict(X_test)

np.mean(predict == y_test)

#######################################################################


############### SERIALIZING THE MODEL ####################
import pickle
file = open('BANK_TRANSACTION_USING_NAIVE_BAYES.pkl', 'wb')
pickle.dump(clf,file,protocol=2)
file.close()
##########################################################
















