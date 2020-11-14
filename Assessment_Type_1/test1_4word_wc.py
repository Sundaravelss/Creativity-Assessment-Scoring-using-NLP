# -*- coding: utf-8 -*-

#1.Four Connected Word Task- From the list of 10 words provided, find as many different groups of 3 words as you can that are somehow 
#associated with a 4th word.

#2.WordChain Task	Starting with the word "chair" please fill-in each of the blanks below with a word that comes to mind based upon
#the previous word in the list. 

#run it with the following command in server:
#python3 test1_4word_wc.py 'limesurveyID' "limesurveySessionID" "UY_userID" "UY_testID"


import argparse
import mysql.connector as mariadb
from scipy import stats
import numpy as np
import enchant
import string
import torch
#pip install --upgrade tensorflow
#pip install --upgrade flair
from flair.data import Sentence
from flair.embeddings import (DocumentPoolEmbeddings,BertEmbeddings,XLNetEmbeddings)
import pickle

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
db_connect_limesurvey = mariadb.connect(host="localhost",user="root",passwd="***",db="limesurvey")
db_connect_understandyourself = mariadb.connect(host="localhost",user="root",passwd="****",db="understandyourself")
cursor_limesurvey = db_connect_limesurvey.cursor()
cursor_understandyourself = db_connect_understandyourself.cursor()

#Get the answers of the 4word task whose survey id is 206
query_column =(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='lime_survey_{lime_id}' AND COLUMN_NAME LIKE '%206%'");
cursor_limesurvey.execute(query_column)
sql_columns=cursor_limesurvey.fetchall()
##print(sql_columns)

#sql_columns has the column names in list of list e.g [[('195891X206X791',)], [('195891X206X792',)]
#looping to get the column name and append it to "columns" list
columns=[]
for col in sql_columns:
    columns.append(col[0])

results=[]
for col in columns:
    #query=("SELECT {} FROM lime_survey_195891 GROUP BY id DESC LIMIT 1".format(col))
    query=(f"SELECT {col} FROM lime_survey_{lime_id} WHERE id={lime_session_id}")
    cursor_limesurvey.execute(query)
    results.append(cursor_limesurvey.fetchall())

#looping to get the answers alone and adding it to the list answers
answers=[]
for index in range(len(results)):
    answers.append(results[index][0][0])
##print(answers)
#so, the final answers looks like [A1', 'A7', 'A3', 'chair', 'A2', 'A8', 'A9', 'office']

#mapping survey drag id to actual word i.e A1 to chimney ,A2 to meeting
mapping={'A1':'chimney','A2':'meeting','A3':'tennis','A4':'money','A5':'cannon','A6':'glass','A7':'round','A8':'camping','A9':'chair','A10':'office'}
#get function in dictionary takes the key as argument and returns the value and the 2nd argument is default value when key is not present
real_answers=[mapping.get(i,i) for i in answers]
#first element in real_answers doesn't have the actual answer ,it is null value in database , so removing it
real_answers=real_answers[1:]
#print(f"all real answers:{real_answers}")

#assigning 60 answers (4*15) in real_answers variable to corresponding word and resp
i=0
for resp in range(1,16):
    for word in range(1,5):
        exec(f"word{word}resp{resp}='{real_answers[i]}'")
        i=i+1

#checking if two word sets are the same and removing the same
for i in range(1,15):
        for j in range(i+1,16):
                if eval(f"word1resp{i}")!=None and eval(f"word1resp{j}")!= None:
                       if (eval(f"word1resp{i}.lower()")==eval(f"word1resp{j}.lower()") and eval(f"word2resp{i}.lower()")==eval(f"word2resp{j}.lower()") and eval(f"word3resp{i}.lower()")==eval(f"word3resp{j}.lower()") and eval(f"word4resp{i}.lower()")==eval(f"word4resp{j}.lower()")):
                                exec(f"word4resp{i}=''")

word4list=[]

#appending all 4th word to list
for i in range(1,16):
        if(eval(f'word4resp{i}') is not None):
                word4list.append(eval(f"word4resp{i}.lower()"))



#to remove punctuation
punctexclude=set(string.punctuation)

#removing empty strings
word4list=list(filter(lambda x:x!='',word4list))
#print("All 4th words: ",word4list)

#checking whether word is in english dictionary by checking the spelling

#using english dictionary en_US
#for french replace "en_US" with "fr" , we can see the list of languages with the command "enchant.list_languages()"
spell = enchant.Dict("en_US")
notjunk=[w for w in word4list if spell.check(w) and w not in punctexclude and not w.isdigit()]
#print("Not junk 4th responses:",notjunk)

#placeholder for the list of words
listofwords=['chimney','meeting','tennis','money','cannon','glass','round','camping','chair','office']

word4=[w for w in notjunk if w not in listofwords]

#counting number of valid answers
count=len(word4)
#print("Number of valid answers:",count)


#removing empty responses and irrelevant junk responses

#valid_responses will have the response set whose 4th word are not junk 
valid_responses=[]
#null_responses will have all the response set whose any of four words are empty (removing set if its left empty or unattended)
null_responses=set()

for resp in range(1,16):
    for word in range(1,5):
        #not(word) returns True if the word is empty
        if eval(f"not word{word}resp{resp}"):
            null_responses.add(resp)
        if word==4:
            #checking if the 4th word is in the filtered non junk valid 4th word list 'word4'
            if eval(f"word{word}resp{resp} in {word4}"):
                valid_responses.append(resp)
#print(f"valid_responses set:{valid_responses}")
#print(f"null_responses set:{null_responses}")

#removing the responses set that are null_responses and keeping only the valid_responses set
final_valid_answers=[]
#looping through 15 response set
for resp in range(1,16):
    #keeping only the set that is in valid_responses and removing the sets that are in null_responses
    if resp in valid_responses and resp not in null_responses:
        for word in range(1,5):
            ##print(resp,word)
            exec(f"final_valid_answers.append(word{word}resp{resp})")
#print(f"final_valid_answers:{final_valid_answers},the length of final_valid_answers:{len(final_valid_answers)}")

#load pretrained embeddings
with open('loaded.embedding','rb') as file:
    embeddings=pickle.load(file)

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

#converting each word into flair acceptable sentence object and then embed the object
final_valid_answers_sent=[]
for i in range(len(final_valid_answers)):
    final_valid_answers_sent.append(Sentence(final_valid_answers[i]))
#embedding the words
embeddings.embed(final_valid_answers_sent)

#comparing(calculating the cosine similarity between) each 3 words in a response set with its 4th word and 
#adding all the cosine similarity scores to get the final score
cosine_score=0
#k is a counter through all valid words
k=1
#j refers to each 3 words 1-3 in a set
for i in range(1,int(len(final_valid_answers)/4)+1):
    for j in range(4):
        ##print(i,j,k)
        #if its the 4th word just continue to next word since only first 3 words to be compared with 4th word in a set
        if k%4==0:
            k+=1
            continue
        else:
            #adding each cosine similarity score between each 3 words in the set with its 4th word
            cosine_score+=cos(final_valid_answers_sent[k-1].embedding,final_valid_answers_sent[(i*4)-1].embedding)
            k+=1
final_score=cosine_score.item()    
#print(f"The Total score is:{cosine_score.item()}") 

def calculatePercentile(raw_score, avg_raw_score, std_deviation):
    '''gets score and returns percentile'''
    z_score = (raw_score - avg_raw_score ) / std_deviation
    percentile = (stats.norm.cdf(z_score) * 100)
    return percentile

avg_raw_score_count=15
std_deviation_count=7
percentile_count=calculatePercentile(count,avg_raw_score_count,std_deviation_count)
percentile_count=round(percentile_count,2)

#print(f'The percentile score for the number of valid answers is {percentile_count}')

avg_raw_score_answer=12
std_deviation_answer=7
percentile_answer=calculatePercentile(final_score,avg_raw_score_answer,std_deviation_answer)
percentile_answer=round(percentile_answer,2)

#print(f'The final percentile score of four connected words task is {percentile_answer}')



#############################################################################################################
####################################--2.WORD CHAIN---##########################################################
#############################################################################################################

#	Starting with the word "chair" please fill-in each of the blanks below with a word that comes to mind based upon
#the previous word in the list.

#Fetch answers from the sql db
query_column =(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='lime_survey_{lime_id}' AND COLUMN_NAME LIKE '%wordchain%'");
cursor_limesurvey.execute(query_column)
sql_columns=cursor_limesurvey.fetchall()

#looping to get the column name and append it to "columns" list
columns=[]
for col in sql_columns:
    columns.append(col[0])

results=[]
for col in columns:
    #query=("SELECT {} FROM lime_survey_195891 GROUP BY id DESC LIMIT 1".format(col))
    query=(f"SELECT {col} FROM lime_survey_{lime_id} WHERE id={lime_session_id}")
    cursor_limesurvey.execute(query)
    results.append(cursor_limesurvey.fetchall())

#looping to get the answers alone and adding it to the list answers
answers=[]
for index in range(len(results)):
    answers.append(results[index][0][0])
#print(answers)

embeddings=DocumentPoolEmbeddings([BertEmbeddings()])
    
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

#converting each word into flair acceptable sentence object and then embed the object
answer_sent=[]
for word in answers:
    answer_sent.append(Sentence(word))
embeddings.embed(answer_sent)

cosine_score=0
for i in range(len(answers)-1):
    cosine_score+=cos(answer_sent[i].embedding,answer_sent[i+1].embedding)
final_score_wc=cosine_score.item()
#print(f"The Total score for wordchain task  is:{cosine_score.item()}") 


avg_raw_score_wc=5
std_deviation_wc=2
percentile=calculatePercentile(final_score_wc,avg_raw_score_wc,std_deviation_wc)
percentile=round(percentile,2)

#print(f'The percentile score for wordchain task is {percentile}')


##################################################
############ Non-reg users result ################
##################################################

if(int(uy_userid) == 0):
    heading = "Thanks for taking the test. Here is your result"
    count='The percentile score of number of valid answers in four connected words task is '+str(percentile_count)
    answer='The percentile score of four connected words task is '+str(percentile_answer)
    wc='The percentile score for wordchain task is '+str(percentile)
    end='"Please register in the website for a detailed report or contact admin@understandyourself.org"'
    message_display=heading+'\n'+count+'\n'+answer+'\n'+wc+'\n'+end

##########################################################################
############## Registered Users Result Display and DB update ############
#########################################################################

###################PLOT################################

elif(int(uy_userid)!=0):

    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    
    
    user_name = ""
    #getting the username from the understandyourself database
    sql_fetchName_UY = ("SELECT UY_user_nicename FROM uy_profile WHERE UY_user_id ='{}'".format(uy_userid))
    cursor_understandyourself.execute(sql_fetchName_UY)
    user_name_value = cursor_understandyourself.fetchall()
    user_name = user_name_value[0][0]
    #print("User Nice Name : ",user_name)
    test_name= ""
    
    #getting the testname from the understandyourself database
    sql_testName_UY = ("SELECT UY_test_name FROM uy_test WHERE UY_test_id ='{}'".format(uy_testid))
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
    ax1.set_title('Percentile score for the number of valid answers in 4word task',fontsize=30)
    ax1.set_xlabel('Percentile',fontsize=30)
    ax1.set_yticklabels([])
    ax1 = plt.plot(x,y, color='coral',linewidth=10)
    ax1 = plt.axvline(percentile_count, x_min, x_max,linewidth=5, label=str(percentile_count)+' Percentile',color='darkblue')
    ax1 = plt.legend(loc="upper right",fontsize=30) 
    
    ax2 = fig.add_subplot(3,2,2)
    ax2.set_title('Percentile score for the correct four connected words',fontsize=30)
    ax2.set_xlabel('Percentile',fontsize=30)
    ax2.set_yticklabels([])
    ax2 = plt.plot(x,y, color='coral',linewidth=10)
    ax2 = plt.axvline(percentile_answer, x_min, x_max,linewidth=5, label=str(percentile_answer) + ' Percentile',color='darkred')
    ax2 = plt.legend(loc="upper right",fontsize=30)
    
    ###################PLOT for word chain task################################
    ax3 = fig.add_subplot(3,2,3)
    ax3.set_title('Percentile score for the WordChain task',fontsize=30)
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
    #print("Score Analysis File: ",score_analysis_hash)
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

#db_connect_wordpress.close()
db_connect_understandyourself.close()
db_connect_limesurvey.close()
