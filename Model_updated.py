#!/usr/bin/env python
# coding: utf-8

# In[354]:


from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import datetime
import pickle


# In[422]:


import numpy as np
import pandas as pd

df=pd.read_csv('model_data_reg_updated.csv')


# In[423]:


df.head()


# In[424]:


df["game_date"]=pd.to_datetime(df["game_date"])


# In[425]:


list=df.values.tolist()


# In[426]:


list_of_tuples=[]
for mini_list in list:
    list_of_tuples.append((mini_list[0],mini_list[1],mini_list[2],mini_list[3],mini_list[4],mini_list[5],mini_list[6],mini_list[7], mini_list[8]))
    


# In[427]:


len(list_of_tuples)


# In[366]:




def lower_bound(array, first, last, value):
    """
    binary search -- return the index of first appearance record that's greater or equal than value
    :param array: the sorted array
    :param first: starting index
    :param last: end index, which is not included in the search
    :param value: target array
    :return: index of the first appearance
    """
    first_ = first
    last_ = last
    while first_ < last_:
        mid = first_ + (last_ - first_) // 2
        if array[mid] < value:
            first_ = mid + 1
        else:
            last_ = mid
    return first_



# In[386]:


def read_data(list_of_tuples):
    """

    :return: a List of (play_id, game_date, quarter_number(range from 1 to 5),
    remaining_quarter_minutes, score_difference_between_the_teams, yard_line(1 to 100),
    down(0 to 4), yards_to_go(0 to 50), scoring_points_of_the_play)
    """
    return list_of_tuples


# In[387]:


def main(list_of_tuples):
        
    data = read_data(list_of_tuples)
    
  
    # sort data by game date
    data = sorted(data, key=lambda x: x[1])

    play_ids = []
    game_dates = []
    dataset = []
    labels = []

    # get the training dataset and labels
    for (play_id, game_date, quarter, quarter_minutes_remaining,
         score_differential, yard_line, down, yards_to_go, drive_points) in data:
        play_ids.append(play_id)
        game_dates.append(game_date)
        dataset.append([quarter, quarter_minutes_remaining,
                        score_differential, yard_line, down, yards_to_go])
        labels.append(drive_points)
        
   
    first_date = game_dates[0]

    xgb_params = {'n_jobs': 10, 'max_depth': 2,
                  'min_child_weight': 10,
                  'tree_method': 'hist', 'n_estimators': 150}
    xgb_model = None
    linear_model = None
    result_str = ''

    # use the data of past 5 years to train the data
    look_back = 365 * 5
    for i, game_date in enumerate(game_dates):
        if (game_date - first_date).days < look_back:
            continue
        start_date = game_date - datetime.timedelta(days=look_back)
        start_index = lower_bound(game_dates, 0, len(game_dates), start_date)
        if None in dataset[i]:
            continue
        if game_dates[i-1].year != game_date.year or xgb_model is None:
            null_data_index = set()
            for j in range(start_index, i):
                if None in dataset[j] or labels[j] is None:
                    null_data_index.add(j)
            train_data = [dataset[j] for j in range(start_index, i) if j not in null_data_index]
            train_labels = [labels[j] for j in range(start_index, i) if j not in null_data_index]
            train_data = np.array(train_data)
            train_labels = np.array(train_labels)
            xgb_model = XGBRegressor(**xgb_params)
            linear_model = LinearRegression(n_jobs=10)
            xgb_model.fit(train_data, train_labels)
            linear_model.fit(train_data, train_labels)
        xgb_score = xgb_model.predict(np.array([dataset[i]]))[0]
        linear_score = linear_model.predict([dataset[i]])[0]
        play_id = play_ids[i]
        game_date = game_dates[i].strftime('%Y-%m-%d')
        result_str += f'{play_id},{game_date},{xgb_score},{linear_score}\n'
        
    return [xgb_model, linear_model]


# In[428]:


results=main(list_of_tuples)


# In[372]:


pickle.dump(results[1], open("final_linear_model7.pickle", "wb"))


# In[373]:


pickle.dump(results[0], open("final_xgb_model7.pickle", "wb"))


# In[374]:


loaded_xgb_model=pickle.load(open('final_xgb_model7.pickle', 'rb'))


# In[375]:


loaded_linear_model=pickle.load(open('final_linear_model7.pickle', 'rb'))


# In[449]:


results[0].predict([[1, 682, .00000001, 43.5, 4,18]])


# In[211]:


results_list=[]
for (play_id, game_date, quarter, quarter_minutes_remaining, score_differential, yard_line, down, distance, drive_points) in list_of_tuples:
    results_list.append((play_id, game_date, quarter, quarter_minutes_remaining, score_differential, yard_line, down, distance, drive_points, loaded_xgb_model.predict([quarter, quarter_minutes_remaining, float(score_differential), yard_line, down, distance]), loaded_linear_model.predict([[quarter, quarter_minutes_remaining, float(score_differential), yard_line, down, distance]])))


# In[181]:


final_list=[]
for (a, b, c, d, e, f, g, h, i, j, k) in results_list:
    final_list.append((a,b,c,d,e,f,g,h,i,j[0], k[0]))


# In[183]:


df1=pd.DataFrame(final_list)


# In[184]:


df1.head()


# In[185]:


df2=df1.rename(columns={0:"play_id", 1:"game_date", 2:"quarter", 3:"game_clock", 4:"score_difference",5:"yardline", 6: "down", 7:"distance", 8:"drive_points",9: "xgb_model_score",10:"linear_model_score" })


# In[188]:


df2.head()


# In[189]:


df2.to_csv("model_data_with_scores3.csv")


# In[ ]:




