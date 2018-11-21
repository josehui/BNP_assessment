import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import pprint
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import gensim
import os.path
import matplotlib
import matplotlib.pyplot as plt
import spacy

#for Docker
import nltk
nltk.download('punkt')

#curent directory
cwd = os.getcwd()


#pretty print stuff
pp = pprint.PrettyPrinter()

#key for newsapi
key = 'd6f155038c014316a4e9622dab7b1a5e'

#Referecne: https://www.youtube.com/watch?v=XyfCVg677Bk&t=860s

class News:
    def __init__(self, cat='industry', country = 'hk', query='US technology news'):
        self.query = query
        self.url = '/https://newsapi.org/v2/everything'

    '''def search (self):
        r = requests.get(url=self.url)
        soup = BeautifulSoup(r.text, 'html.parser')
        s_summary = soup.find_all('div', {'class': 'st'})
        for s in s_summary:
            print (s.text)'''

    def search_api(self):
        articles=[]
        page = 1
        url = 'https://newsapi.org/v2/everything'
        while True:
            # reference: https://newsapi.org/docs
            r = requests.get(url, params={'apiKey': key,'q': self.query,'pageSize':100, 'language':'en', 'page':page, 'sortby': 'relevancy'})
            r_j = r.json()
            if r_j['status']=='error' or r_j['articles']==[]:
                print ('All atricles retreived')
                break
            #print ('page: ', page)
            #print (r_j['articles'][0]['title'])
            #print (r_j['articles'][0]['publishedAt'])
            articles += r.json()['articles']
            page+=1
        #print (len(articles))
        df = pd.DataFrame(articles)
        now = datetime.now().strftime("%Y-%m-%d")
        df.to_csv(cwd+'/Articles/{}_{}_articles.csv'.format(now, self.query))
###
    def similarity_analyze_tfidf(self):
        a_df = pd.read_csv(cwd+'/Articles/2018-11-21_'+self.query+'_articles.csv',index_col=0)
        #Data cleaning
        a_df.dropna(axis=0, subset=['title', 'content', 'publishedAt'], inplace=True)
        a_df.sort_values('publishedAt', inplace=True)
        #add column to represent count of each topic (before grouping)
        a_df['count']=1
        a_df['publishedAt'] = pd.to_datetime(a_df['publishedAt'])
        a_df.reset_index(drop=True, inplace=True)
        output_df = a_df.copy(True)
        exclude =[]
        grp = {}
        #Each article is a grp of its own in the beginning
        for ix in a_df.index:
            grp[ix] = ix
        #Evaluate each article
        for index, row in a_df.iterrows():
            #continue if article already removed
            if index in exclude:
                continue
            date = row['publishedAt']
            start_date = date - timedelta(days=1)
            end_date = date+ timedelta(days=1)
            print ('date:{}, start_date: {}, end_date: {}'.format(date,start_date,end_date))
            #Filter articles within +- 1 day
            sub_df = a_df[(a_df['publishedAt'] >= start_date) & (a_df['publishedAt'] <= end_date)]
            sub_titles = sub_df['title'].tolist()
            sub_contents = [c[:-10] for c in sub_df['content'].tolist()]

            #Tokenize the artcle content
            sub_df['token_list'] = [[w.lower() for w in word_tokenize(t)] for t in sub_df['content']]
            #Create dictionary with the tokens
            word_dict = gensim.corpora.Dictionary(sub_df['token_list'])
            #print("Number of words in dictionary:",len(word_dict))
                
            #Create corpus: lists the number of times each word occurs a title
            corpus = [word_dict.doc2bow(token) for token in sub_df['token_list']]
            #print(corpus)

            #Create tf-idf model
            tf_idf = gensim.models.TfidfModel(corpus)

            sims = gensim.similarities.Similarity('',tf_idf[corpus],num_features=len(word_dict))

            query_doc = [w.lower() for w in word_tokenize(row['content'])]
            query_doc_bow = word_dict.doc2bow(query_doc)
            query_doc_tf_idf = tf_idf[query_doc_bow]
            sub_df['sims_scores'] = sims[query_doc_tf_idf]
            print ('Title: ', row['title'])
            #print ('Content: ', row['content'])
            m_ind = sub_df[sub_df['sims_scores']>0.25]
            #display(m_ind)
            idx = m_ind.index
            print (idx)
            if len(idx) > 1:
                k = grp[idx[0]]
                idx = idx[1:]
                #remove exlcuded articles
                idx = [x for x in idx if x not in exclude]
                #group articles
                for ix in idx:
                    grp[ix]=k
                output_df.at[k, 'count'] += len(idx)
                output_df.drop(idx, inplace=True)
                exclude += idx
        print ('Number of unique news event: ', len(output_df.index))
        print ('Averge number of report per event: ', output_df['count'].mean())
        now = datetime.now().strftime("%Y-%m-%d")
        output_df.to_csv(cwd+'/Output/{}_{}_output(tf-idf).csv'.format(now, self.query))

    def similarity_analyze_vector(self):
        nlp = spacy.load('en')

        a_df = pd.read_csv(cwd+'/Articles/2018-11-18_hsbc_articles.csv',index_col=0)
        #Data cleaning
        a_df.dropna(axis=0, subset=['title', 'content', 'publishedAt'], inplace=True)
        a_df.sort_values('publishedAt', inplace=True)
        #add column to represent count of each topic (before grouping)
        a_df['count']=1
        a_df['publishedAt'] = pd.to_datetime(a_df['publishedAt'])
        a_df.reset_index(drop=True, inplace=True)
        output_df = a_df.copy(True)
        exclude = []
        grp = {}
        #Each article is a grp of its own in the beginning
        for ix in a_df.index:
            grp[ix] = ix
        #display(output_df.head(10))
        #Evaluate each article
        for index, row in a_df.iterrows():
            #continue if article already removed
            if index in exclude:
                continue
            date = row['publishedAt']
            start_date = date - timedelta(days=1)
            end_date = date+ timedelta(days=1)
            print ('date:{}, start_date: {}, end_date: {}'.format(date,start_date,end_date))
            #Filter articles within +- 1 day
            sub_df = a_df[(a_df['publishedAt'] >= start_date) & (a_df['publishedAt'] <= end_date)]
            
            #Calculate similarity based on pre-trained word vector from Spacy
            #https://spacy.io/usage/vectors-similarity
            sub_df['similarity'] = [nlp(row['title']).similarity(nlp(r['title'])) for i, r in sub_df.iterrows()]
            print ('Title: ', row['title'])
            print ('Content: ', row['content'])
            #
            m_ind = sub_df[sub_df['similarity']>0.85]
            idx = m_ind.index
            if len(idx) > 1:
                k = grp[idx[0]]
                idx = idx[1:]
                #remove exlcuded articles
                idx = [x for x in idx if x not in exclude]
                #group articles
                for ix in idx:
                    grp[ix]=k
                output_df.at[k, 'count'] += len(idx)
                output_df.drop(idx, inplace=True)
                exclude += idx
        print ('Number of unique news event: ', len(output_df.index))
        print ('Averge number of report per event: ', output_df['count'].mean())
        now = datetime.now().strftime("%Y-%m-%d")
        output_df.to_csv(cwd+'/Output/{}_{}_output(vector_model).csv'.format(now, self.query))

class Stock_Analyize:
    def __init__(self, filename='2018-11-21_US technology news_output(tf-idf).csv'):
        self.filename = filename


    def US_Technology_analyze(self):

        # ## US technology sector

        # In[336]:
        s_df = pd.read_csv(cwd+'/Output/'+filename,index_col=0)
        s_df['publishedAt'] = pd.to_datetime(s_df['publishedAt'])
        print (s_df.shape)
        #Exclude articles with only one mentioning
        input_df = s_df[s_df['count']>1]
        input_df.sort_values('count',ascending=False, inplace=True)
        input_df.reset_index(drop=True, inplace=True)
        print (input_df.shape)


        # In[337]:


        #distribution of no. of articles for a single topic
        input_df.plot(y='count', use_index=True)


        # ## a) Impact on NASDAQ 100 Technology Sector (^NDXT)

        # In[338]:


        #Data from yahoo finance
        ndxt_df = pd.read_csv(cwd+'/Data/17-18_NDXT.csv')
        ndxt_df.dropna(axis=0, inplace=True)
        ndxt_df['Date'] = pd.to_datetime(ndxt_df['Date'])
        #add closing hour: 4pm
        ndxt_df['Date'] = ndxt_df['Date'] + timedelta(hours=16)
        ndxt_df['Change'] = ndxt_df['Close'] - ndxt_df.shift(1)['Close']
        ndxt_df['Change_%'] = ndxt_df['Close']/ndxt_df.shift(1)['Close']-1
        ndxt_df['Day_Change'] = ndxt_df['Close'] - ndxt_df['Open']
        ndxt_df['Day_Change(%)'] = (ndxt_df['Close']/ndxt_df['Open']-1)*100


        # In[339]:


        start = input_df['publishedAt'].min()
        end = input_df['publishedAt'].max()
        print (start, end)


        # In[340]:


        NT_df = ndxt_df[(ndxt_df['Date']>=start) & (ndxt_df['Date'] <= end)]
        NT_df.reset_index(drop=True,inplace=True)
        NT_df


        # In[341]:


        #Get closest trading date according to publish date
        #Reference:  https://stackoverflow.com/questions/39105282/how-to-find-min-value-of-another-column-greater-than-current-column-pandas
        def NT_findMin_idx(x):
            larger = NT_df[NT_df['Date']>x]['Date']
            if len(larger) !=0:
                return larger.idxmin()
            else:
                return np.nan #if no article is too new and no market data avliable yet


        # In[342]:


        for i, r in input_df.iterrows():
            Min_idx = NT_findMin_idx(r['publishedAt'])
            if np.isnan(Min_idx): continue
            input_df.at[i,'Trading_day'] = NT_df.loc[Min_idx,'Date']
            # Define impact as the magntudde of change
            input_df.at[i,'Impact'] = NT_df.loc[Min_idx,'Change']


        # In[343]:


        #rank by change magnitude + number of mentioning
        input_df['abs'] = input_df['Impact'].abs()
        ranked_df = input_df.sort_values(['abs', 'count'], ascending = False)
        ranked_df.drop('abs', axis=1, inplace=True)
        ranked_df.reset_index(drop=True, inplace=True)
        ranked_df.head()


        # In[344]:


        matplotlib.rcParams.update({'font.size': 15})

        plt.figure(figsize=(20,10))
        plt.plot(NT_df['Date'],NT_df['Close'])
        plt.ylabel('some numbers')
        # draw vertical lines for top topics
        for i, r in ranked_df.drop_duplicates('Trading_day').head(10).iterrows():
            d = r['publishedAt']
            plt.axvline(x=d , color='r')
            #label
            plt.text(d, 3900, r['title'][:40]+'...', rotation=90, verticalalignment='center')
        plt.show()


        # ## Ouput ranking 

        # In[345]:


        ranked_df.index += 1 
        ranked_df.index.names = ['Rank']
        ranked_df.to_excel(cwd+'/Result/2018-11-21_NDXT_impact.xlsx')


        # ## b) Impact on NASDAQ Composite (^IXIC)

        # In[346]:


        #Data from yahoo finance
        ixic_df = pd.read_csv(cwd+'/Data/17-18_IXIC.csv')
        ixic_df.dropna(axis=0, inplace=True)
        ixic_df['Date'] = pd.to_datetime(ixic_df['Date'])
        #add closing hour: 4pm
        ixic_df['Date'] = ixic_df['Date'] + timedelta(hours=16)
        ixic_df['Change'] = ixic_df['Close'] - ixic_df.shift(1)['Close']
        ixic_df['Change_%'] = ixic_df['Close']/ixic_df.shift(1)['Close']-1
        ixic_df['Day_Change'] = ixic_df['Close'] - ixic_df['Open']
        ixic_df['Day_Change(%)'] = (ixic_df['Close']/ixic_df['Open']-1)*100


        # In[347]:


        start = input_df['publishedAt'].min()
        end = input_df['publishedAt'].max()
        print (start, end)


        # In[348]:


        IC_df = ixic_df[(ixic_df['Date']>=start) & (ixic_df['Date'] <= end)]
        IC_df.reset_index(drop=True,inplace=True)
        IC_df


        # In[349]:


        #Get closest trading date according to publish date
        #Reference:  https://stackoverflow.com/questions/39105282/how-to-find-min-value-of-another-column-greater-than-curreIC-column-pandas
        def IC_findMin_idx(x):
            larger = IC_df[IC_df['Date']>x]['Date']
            if len(larger) !=0:
                return larger.idxmin()
            else:
                return np.nan #if no article is too new and no market data avliable yet


        # In[350]:


        for i, r in input_df.iterrows():
            Min_idx = IC_findMin_idx(r['publishedAt'])
            if np.isnan(Min_idx): continue
            input_df.at[i,'Trading_day'] = IC_df.loc[Min_idx,'Date']
            # Define impact as the magICudde of change
            input_df.at[i,'Impact'] = IC_df.loc[Min_idx,'Change']


        # In[351]:


        #rank by change magnitude + number of mentioning
        input_df['abs'] = input_df['Impact'].abs()
        ranked_df = input_df.sort_values(['abs', 'count'], ascending = False)
        ranked_df.drop('abs', axis=1, inplace=True)
        ranked_df.reset_index(drop=True, inplace=True)
        ranked_df.head()


        # In[352]:


        matplotlib.rcParams.update({'font.size': 15})

        plt.figure(figsize=(20,10))
        plt.plot(IC_df['Date'],IC_df['Close'])
        plt.ylabel('some numbers')
        # draw vertical lines for top topics
        for i, r in ranked_df.drop_duplicates('Trading_day').head(10).iterrows():
            d = r['publishedAt']
            plt.axvline(x=d , color='r')
            #label 
            plt.text(d, 7200, r['title'][:40]+'...', rotation=90, verticalalignment='center')
        plt.show()


        # ## Ouput ranking 

        # In[353]:


        ranked_df.index += 1 
        ranked_df.index.names = ['Rank']
        print (ranked_df)
        ranked_df.to_excel(cwd+'/Result/2018-11-21_IXIC_impact.xlsx')

    def HK_analyze(self):

        # ## HK region

        # In[389]:


        s_df = pd.read_csv(cwd+'/Output/2018-11-21_Hong Kong news_output(tf-idf).csv',index_col=0)
        s_df['publishedAt'] = pd.to_datetime(s_df['publishedAt'])
        print (s_df.shape)
        #Exclude articles with only one mentioning
        input_df = s_df[s_df['count']>1]
        input_df.sort_values('count',ascending=False, inplace=True)
        input_df.reset_index(drop=True, inplace=True)
        print (input_df.shape)


        # In[390]:


        #distribution of no. of articles for a single topic
        input_df.plot(y='count', use_index=True)


        # ## Impact on Hang Seng Index (^HSI)

        # In[391]:


        #Data from yahoo finance
        hsi_df = pd.read_csv(cwd+'/Data/17-18_HSI.csv')
        hsi_df.dropna(axis=0, inplace=True)
        hsi_df['Date'] = pd.to_datetime(hsi_df['Date'])
        #add closing hour: 4pm
        hsi_df['Date'] = hsi_df['Date'] + timedelta(hours=16)
        hsi_df['Change'] = hsi_df['Close'] - hsi_df.shift(1)['Close']
        hsi_df['Change_%'] = hsi_df['Close']/hsi_df.shift(1)['Close']-1
        hsi_df['Day_Change'] = hsi_df['Close'] - hsi_df['Open']
        hsi_df['Day_Change(%)'] = (hsi_df['Close']/hsi_df['Open']-1)*100


        # In[392]:


        start = input_df['publishedAt'].min()
        end = input_df['publishedAt'].max()
        print (start, end)


        # In[393]:


        H_df = hsi_df[(hsi_df['Date']>=start) & (hsi_df['Date'] <= end)]
        H_df.reset_index(drop=True,inplace=True)
        H_df


        # In[394]:


        #Get closest trading date according to publish date
        #Reference:  https://stackoverflow.com/questions/39105282/how-to-find-min-value-of-another-column-greater-than-current-column-pandas
        def H_findMin_idx(x):
            larger = H_df[H_df['Date']>x]['Date']
            if len(larger) !=0:
                return larger.idxmin()
            else:
                return np.nan #if no article is too new and no market data avliable yet


        # In[395]:


        for i, r in input_df.iterrows():
            Min_idx = H_findMin_idx(r['publishedAt'])
            if np.isnan(Min_idx): continue
            input_df.at[i,'Trading_day'] = H_df.loc[Min_idx,'Date']
            # Define impact as the magntudde of change
            input_df.at[i,'Impact'] = H_df.loc[Min_idx,'Change']


        # In[396]:


        #rank by change magnitude + number of mentioning
        input_df['abs'] = input_df['Impact'].abs()
        ranked_df = input_df.sort_values(['abs', 'count'], ascending = False)
        ranked_df.drop('abs', axis=1, inplace=True)
        ranked_df.reset_index(drop=True, inplace=True)
        ranked_df.head()


        # In[397]:


        matplotlib.rcParams.update({'font.size': 15})

        plt.figure(figsize=(20,10))
        plt.plot(H_df['Date'],H_df['Close'])
        plt.ylabel('some numbers')
        # draw vertical lines for top topics
        for i, r in ranked_df.drop_duplicates('Trading_day').head(10).iterrows():
            d = r['publishedAt']
            plt.axvline(x=d , color='r')
            #label
            plt.text(d, 25500, r['title'][:40]+'...', rotation=90, verticalalignment='center')
        plt.show()

        ## Ouput ranking 
        hsi_df = pd.read_csv(cwd+'/Data/17-18_HSI.csv')
        hsi_df.dropna(axis=0, inplace=True)
        hsi_df['Date'] = pd.to_datetime(hsi_df['Date'])
        #add closing hour: 4pm
        hsi_df['Date'] = hsi_df['Date'] + timedelta(hours=16)
        hsi_df['Change'] = hsi_df['Close'] - hsi_df.shift(1)['Close']
        hsi_df['Change_%'] = hsi_df['Close']/hsi_df.shift(1)['Close']-1
        hsi_df['Day_Change'] = hsi_df['Close'] - hsi_df['Open']
        hsi_df['Day_Change_%'] = hsi_df['Close']/hsi_df['Open']-1


        # ## Ouput ranking 

        # In[398]:


        ranked_df.index += 1 
        ranked_df.index.names = ['Rank']
        print (ranked_df)
        ranked_df.to_excel(cwd+'/Result/2018-11-21_HSI_impact.xlsx')




if __name__ == '__main__':
    print(cwd)
    choice= '0'
    while choice != 5:
        print(30 * "-" , "MENU" , 30 * "-")
        print("1: Serach for news according to industry/ region")
        print("2: Count unique news event (tf-idf model)")
        print("3: Count unique news event (word model)") # takes a long time, not recommended
        print("4: Rank impact on stock price") #using ouput file from 2 or 3
        print("5: Quit")
        choice = input ("Please make a choice: ")
        print (choice)
        if choice=='1':   
            query = input('Which industry are you interested in?')  
            a = News(query)
        elif choice=='2':
            a = News()
            a.similarity_analyze_tfidf()
        elif choice=='3':
            a = News()
            a.similarity_analyze_vector()
        elif choice=='4':
            print(30 * "-" , "OPTION" , 30 * "-")
            print("1: Rank impact on US technology sector")
            print("2: Rank impact on Hong Kong region")
            option = input("Please select one of the above")
            if option=='1':   
                filename = '2018-11-21_US technology news_output(tf-idf).csv'
                s = Stock_Analyize(filename)
                s.US_Technology_analyze()
            elif option=='2':
                filename = '2018-11-21_Hong ()Kong news_output(tf-idf).csv'
                s = Stock_Analyize(filename)
                s.HK_analyze()
        elif choice=='5':
            print ("Quiting...")
        else:
            # Any integer inputs other than values 1-5 we print an error message
            choice = input("Wrong option selection. Enter any key to try again..")
            
        
        
    #a.analyze2()


