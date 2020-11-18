
# -*- coding: utf-8 -*-

#1.Image Perception Task- From the list of 10 words provided, find as many different groups of 3 words as you can that are somehow 
#associated with a 4th word.

#2.Wordassociation- The word_association task has 29 different words and the user has to give the related word for each given word and the 
#score is calculated based on the relation between the 2 words(given_word and user_word)


#run it with the following command in server:
#python3 test2_image_wa.py 'limesurveyID' "limesurveySessionID" "UY_userID" "UY_testID"


import argparse
import mysql.connector as mariadb
from scipy import stats
import numpy as np
import nltk
import ssl

#argparse to get the arguments from the command line (which was passed from php code)
parser = argparse.ArgumentParser(description="Script to calculate the score for nlp related surveys")
parser.add_argument("limesurveyID", type=str, help="Give the limesurvey id")
parser.add_argument("limesurveySessionID", type=str, help="Give the limesurvey session id")
parser.add_argument("UY_userID", type=str, help="Give the UY User id corresponding to ultimate member profile")
parser.add_argument("UY_testID", type=str, help= "Give the UY test id corrsponding to Limesurvey")

args = parser.parse_args()

lime_id=args.limesurveyID
lime_session_id=args.limesurveySessionID
uy_userid=args.UY_userID
uy_testid=args.UY_testID


# Open database connection
db_connect_limesurvey = mariadb.connect(host="localhost",user="root",passwd="BTMPsycho2019",db="limesurvey")
db_connect_understandyourself = mariadb.connect(host="localhost",user="root",passwd="BTMPsycho2019",db="understandyourself")
cursor_limesurvey = db_connect_limesurvey.cursor()
cursor_understandyourself = db_connect_understandyourself.cursor()

sql = (f"SELECT * FROM lime_survey_{lime_id} WHERE id={lime_session_id}")
# Execute the SQL command
cursor_limesurvey.execute(sql)
# Fetch all the rows in a list of lists.
results = cursor_limesurvey.fetchall()
answer=[]
#getting the answers of image perception task
for i in range(62,len(results[0])-8):
    answer.append(results[0][i].lower())     #adding all the responses to the answer


#download the nltk wordnet library
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
   pass
else:
   ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')


bird=[answer[i] for i in range(15)]
cloud=[answer[i] for i in range(15,30)]
cello=[answer[i] for i in range(30,45)]

#get the synonym of  words of first image
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
syn_bird = []
for syn in wordnet.synsets("bird"):
    for lm in syn.lemmas():
             syn_bird.append(lm.name())#adding into synonyms
exc_syn_bird=list(set(syn_bird))
print("Synonym of bird:",exc_syn_bird)


syn_bird=['wench', 'skirt', 'hoot', 'raspberry', 'boo', 'doll', 'birdie', 'razzing', 'birdwatch', 'chick', 'snort', 'shuttlecock', 'hiss', 'shuttle', 'razz', 'bird', 'fowl', 'dame', 'Bronx_cheer']
syn_cloud=['mist', 'corrupt', 'befog', 'fog', 'obscure', 'defile', 'overcast', 'becloud', 'taint', 'swarm', 'obnubilate', 'mottle', 'sully', 'cloud', 'haze_over', 'dapple']
syn_cello=['cello', 'violoncello']


#get the synonyms of french word
import nltk 
nltk.download('omw')
from nltk.corpus import wordnet
syn_nouveau=[]
#getting synonyms of word "nouveau" from wordnet 'fra'
for synset in wordnet.synsets('nouveau',lang='fra'):
    #getting the synonym words from the synsets
    for lm in synset.lemma_names('fra'):
        syn_nouveau.append(lm)
syn_nouveau=set(syn_nouveau)



#synonyms list of final answer for 12 image task
syn_glove=['mitt', 'baseball mitt', 'boxing glove', 'baseball glove', 'glove']
syn_dolphin=['mahimahi', 'dolphinfish', 'dolphin']
syn_sandglass=['sandglass']

#Remove punctuation
import string
punctexclude=set(string.punctuation)

import enchant
#spell check to check if the word is meaningful english word
#for french replace "en_US" with "fr" , we can see the list of languages with the command "enchant.list_languages()"
spell=enchant.Dict('en_US')

#################################---bird----#######################################

 #include in this list what are all the words that should be excluded for bird image 
exclude_bird=['','bird']

for index,word in enumerate(bird):
    if word not in exclude_bird and word not in punctexclude and word not in string.digits:
        if spell.check(word) and word not in syn_bird:
            birdguess=index+1       #birdguess score is the first non junk word which is between 2 to 15, 15 for the last response
            #print("First relevant guess word:",word)
            break
    else:
        birdguess=16                #if haven't answered anything meaningful

#print("Index of first relevant guess word for bird:",birdguess)    


#include all the alternative words of the correct answer in this list
rightans_bird=list(syn_glove)+['a glove','building','the glove','cottage','a cottage']


for index,word in enumerate(bird):
    if word not in exclude_bird and word not in punctexclude and word not in string.digits:
        if spell.check(word) and word not in syn_bird:
            if word in rightans_bird:
                birdcorrect=index+1      #birdcorrect score is the index of correct answer which between 2 to 15, 15 for the last response
                #print("Correct guess:",word)
                break
    else:
        birdcorrect=15
#print("Correct guess index",birdcorrect)  


#################################---cloud----#######################################

exclude_cloud=['','cloud']

for index,word in enumerate(cloud):
    if word not in exclude_cloud and word not in punctexclude and word not in string.digits:
        if spell.check(word) and word not in syn_cloud:
            cloudguess=index+1      
            break
    else:
        cloudguess=16                

#print("Index of first relevant guess word for cloud:",cloudguess)   

rightans_cloud=list(syn_dolphin)+['a glove','building','the glove','cottage','a cottage']


for index,word in enumerate(cloud):
    if word not in exclude_cloud and word not in punctexclude and word not in string.digits:
        if spell.check(word) and word not in syn_cloud:
            if word in rightans_cloud:
                cloudcorrect=index+1    
                break
    else:
        cloudcorrect=15
#print("Correct guess index",cloudcorrect)  

#################################---cello----#######################################
 
exclude_cello=['','cello']

for index,word in enumerate(cello):
    if word not in exclude_cello and word not in punctexclude and word not in string.digits:
        if spell.check(word) and word not in syn_cello:
            celloguess=index+1    
            break
    else:
        celloguess=16                

print("Index of first relevant guess word for cello:",celloguess)   

rightans_cello=list(syn_sandglass)+['a glove','building','the glove','cottage','a cottage']

for index,word in enumerate(cello):
    if word not in exclude_cello and word not in punctexclude and word not in string.digits:
        if spell.check(word) and word not in syn_cello:
            if word in rightans_cello:
                cellocorrect=index+1  
                break
    else:
        cellocorrect=15
print("Correct guess index",cellocorrect)  

morphtotalguess=(15-birdguess) +(15-cloudguess)+(15-celloguess)
morphtotalcorrect=(15-birdcorrect)+(15-cloudcorrect)+(15-cellocorrect)

def calculatePercentile(raw_score, avg_raw_score, std_deviation):
    z_score = (raw_score - avg_raw_score ) / std_deviation
    percentile = (stats.norm.cdf(z_score) * 100)
    return percentile

avg_raw_score_guess=15
std_deviation_guess=7
percentile_guess=calculatePercentile(morphtotalguess,avg_raw_score_guess,std_deviation_guess)
percentile_guess=round(percentile_guess,2)

print(f'The percentile score of morphtotalguess is {percentile_guess}')

avg_raw_score_correct=9
std_deviation_correct=3
percentile_correct=calculatePercentile(morphtotalcorrect,avg_raw_score_correct,std_deviation_correct)
percentile_correct=round(percentile_correct,2)

print(f'The percentile score of morphtotalcorrect is {percentile_correct}')

#############################################################################################################
####################################--2.WORD ASSOCIATION---##################################################
#############################################################################################################

#The word_association task has 29 different words and the user has to give the related word for each given word and the 
#score is calculated based on the relation between the 2 words(given_word and user_word)

import torch

from flair.data import Sentence

from flair.embeddings import (DocumentPoolEmbeddings,BertEmbeddings,XLNetEmbeddings)


sql = (f"SELECT * FROM lime_survey_{lime_id} WHERE id={lime_session_id}")
# Execute the SQL command
cursor_limesurvey.execute(sql)
# Fetch all the rows in a list of lists.
results = cursor_limesurvey.fetchall()
##print(results)
answer=[]
#the actual answer starts from 33rd column of the database and before that we have the information of user and
#other question answers and so the answer for word association starts from 33rd column and ending at 62nd column
for i in range(33,62):
    answer.append(results[0][i].lower())     #adding all the responses to the answer
#print(answer)

import pickle

with open('/var/www/html/wordpress/wp-content/custom/python_scripts/given_words.sentence','rb') as file:
    given_words_sent=pickle.load(file)


with open('/var/www/html/wordpress/wp-content/custom/python_scripts/loaded.embedding','rb') as file2:
    embeddings=pickle.load(file2)
    
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

#converting each word into flair acceptable sentence object and then embed the object
answer_sent=[]
for word in answer:
    answer_sent.append(Sentence(word))
embeddings.embed(answer_sent)

cosine_score=0
for i in range(29):
    cosine_score+=cos(given_words_sent[i].embedding,answer_sent[i].embedding)
final_score=cosine_score.item()
print(f"The Total score for word association task is:{cosine_score.item()}")

avg_raw_score_ass=15
std_deviation_ass=7
percentile=calculatePercentile(final_score,avg_raw_score_ass,std_deviation_ass)
percentile=round(percentile,2)

print(f'The percentile score for word association task is {percentile}')


##################################################
############ Non-reg users result ################
##################################################

if(int(uy_userid) == 0):
    heading = "Thanks for taking the test. Here is your result"
    guess='The percentile score for the total first relevant guesses for the Image Perception Task is '+str(percentile_guess)
    answer='The percentile score for the total correct guesses for the Image Perception Task is '+str(percentile_correct)
    wa='The percentile score for word association task is '+str(percentile)
    end='"Please register in the website for a detailed report or contact admin@understandyourself.org"'
    message_display=heading+'\n'+guess+'\n'+answer+'\n'+wa+'\n'+end

##########################################################################
############## Registered Users Result Display and DB update ############
#########################################################################


###################PLOT for image task################################

elif(int(uy_userid)!=0):
        
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    
    user_name = ""
    #getting the username from the understandyourself database
    sql_fetchName_UY = ("SELECT UY_user_nicename FROM uy_profile WHERE UY_user_id ='{}'".format(uy_userid))
    ##print(sql_fetchName_UY)
    cursor_understandyourself.execute(sql_fetchName_UY)
    user_name_value = cursor_understandyourself.fetchall()
    user_name = user_name_value[0][0]
    #print("User Nice Name : ",user_name)
    
    test_name= ""
    #getting the testname from the understandyourself database
    sql_testName_UY = ("SELECT UY_test_name FROM uy_test WHERE UY_test_id ='{}'".format(uy_testid))
    ##print(sql_testName_UY)
    cursor_understandyourself.execute(sql_testName_UY)
    test_name_value = cursor_understandyourself.fetchall()
    test_name = test_name_value[0][0]
    #print("Test Taken : ",test_name)
    
    x_min = 0
    x_max = 100
    
    mean = 50
    std = 20
    
    #to draw percentile line
    x = np.linspace(x_min, x_max, 200)
    y = stats.norm.pdf(x,mean,std)
    
    #setting the plot size and font size and labels
    fig = plt.figure(figsize=(30,40 ),dpi=80)
    fig.suptitle('Profile Analysis', fontsize=75,color='brown')
    fig.text(.20,.93,'User: '+user_name, fontsize=30,color='darkblue')
    fig.text(.70,.93,'Test: '+test_name, fontsize=30,color='darkblue')
    
    ax1 = fig.add_subplot(3,2,1)
    
    ax1.set_title('Percentile score for the total first relevant guesses for the Image Perception Task',fontsize=30)
    ax1.set_xlabel('Percentile',fontsize=30)
    ax1.set_yticklabels([])
    
    ax1 = plt.plot(x,y, color='coral',linewidth=10)
    
    ax1 = plt.axvline(percentile_guess, x_min, x_max,linewidth=5, label=str(percentile_guess)+' Percentile',color='darkblue')
    ax1 = plt.legend(loc="upper right",fontsize=30)
    
    ax2 = fig.add_subplot(3,2,2)
    
    ax2.set_title('Percentile score for the total correct guesses for the Image Perception Task ',fontsize=30)
    ax2.set_xlabel('Percentile',fontsize=30)
    ax2.set_yticklabels([])
    
    ax2 = plt.plot(x,y, color='coral',linewidth=10)
    
    ax2 = plt.axvline(percentile_correct, x_min, x_max,linewidth=5, label=str(percentile_correct) + ' Percentile',color='darkred')
    
    ax2 = plt.legend(loc="upper right",fontsize=30)
    
    ###################PLOT for word association task################################
    
    ax3 = fig.add_subplot(3,2,3)
    
    ax3.set_title('Percentile score for the Word Association task',fontsize=30)
    ax3.set_xlabel('Percentile',fontsize=30)
    ax3.set_yticklabels([])
    
    ax3 = plt.plot(x,y, color='coral',linewidth=10)
    
    ax3 = plt.axvline(percentile, x_min, x_max,linewidth=5, label=str(percentile) + ' Percentile',color='darkred')
    
    ax3 = plt.legend(loc="upper right",fontsize=30)
    
     ###################################################################################################
    ################ For Generating Hash Code for Score Analysis image file############################
    ###################################################################################################
    
    import hashlib
    score_analysis = user_name+test_name
    score_analysis_hash = hashlib.sha1(score_analysis.encode("UTF-8")).hexdigest()+".png"
    
    ###################################################################################################
    #################### Score Analysis image file is stored in the below path ########################
    ###################################################################################################
    
    output_fig_path = os.path.join('/var/www/html/wordpress/wp-content/uploads/scores/'+score_analysis_hash)
    plt.savefig(output_fig_path,dpi=80)
    #plt.show()
    
    ###################################################################################################
    ################### Updating UY DB with the Score Analysis image file name ########################
    ###################################################################################################
    
    sql_insert_UY = "INSERT INTO uy_profile_meta (UY_user_id, UY_meta_key, UY_meta_value) VALUES (%s, %s, %s)"
    val_insert_UY = ('{}'.format(uy_userid), "test_completed_{}_scores".format(uy_testid), score_analysis_hash)
    cursor_understandyourself.execute(sql_insert_UY, val_insert_UY)
    
    db_connect_understandyourself.commit()
    
    message_display='User score calculated'

print(message_display)

###################################################################################################
################################# disconnect from server ##########################################
###################################################################################################

db_connect_understandyourself.close()
db_connect_limesurvey.close()

