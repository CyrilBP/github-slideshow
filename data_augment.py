# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:57:47 2020

@author: Cyril Bosse-Platiere
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
 
def Momentum(data, indices, days1, day_duel, day_surface):
    
    exiting_columns = list(data.columns)
    
    data[["Player1", "Player2"]] = data.loc[:,["Winner", "Loser"]]
    neutral = [0] * data.shape[0]
    
    #dealing with momentum over last set of days
    dic1={('pc_win1_'+str(d)):neutral for d in days1}
    dic2={('pc_win2_'+str(d)):neutral for d in days1}   
    dic ={**dic1, **dic2}
    name_dic = list(dic.keys())
    
    #dealing with momentum per surface
    surfaces = list(set(data.Surface))
    dic_surface1 = {('pc_win1_' + s):neutral for s in surfaces}
    dic_surface2 = {('pc_win2_' + s):neutral for s in surfaces}
    dic_surface = {**dic_surface1, **dic_surface2}
    name_dic_surface = list(dic_surface.keys())
    

    data = pd.concat([data, pd.DataFrame(dic), pd.DataFrame(dic_surface),
                      pd.DataFrame( {'Duel1':neutral, 'Duel2':neutral} )], axis = 1)
    data = data.loc[:,["Player1", "Player2"] + name_dic + name_dic_surface + ["Duel1", "Duel2"] +  exiting_columns]
    
    for i, match_index in enumerate(tqdm(indices)):
        if i%5000 == 0:
            print(i)
        match = data.iloc[match_index,:]
        
        momentum1, momentum2 = [], []
        for d in days1:
            past_matches = data[(data.Date<match.Date)&(data.Date>=match.Date-timedelta(days=d))]
            momentum1 += [Player_Momentum(1,match,past_matches)-0.5]
            momentum2 += [Player_Momentum(2,match,past_matches)-0.5]
        momentum = momentum1 + momentum2
        data.loc[match_index, name_dic] = momentum
        
        momentum_surface1, momentum_surface2 = [], []
        for s in surfaces:
            past_matches = data[(data.Date<match.Date)&(data.Date>=match.Date-timedelta(days=day_surface)) & (data.Surface==s)]
            momentum_surface1 += [Player_Momentum(1,match,past_matches)-0.5]
            momentum_surface2 += [Player_Momentum(2,match,past_matches)-0.5]
        momentum_surface = momentum_surface1 + momentum_surface2
        data.loc[match_index, name_dic_surface] = momentum_surface


        past_matches = data[(data.Date<match.Date)&(data.Date>=match.Date-timedelta(days=day_duel))]
        face_to_face_1 = FacetoFace(1, match, past_matches)-0.5
        
        data.loc[match_index,["Duel1", "Duel2"]] = [face_to_face_1, - face_to_face_1]
         
    return data
    

def Player_Momentum(k, match, past_matches):
    player = match.Winner if k==1 else match.Loser
    ##### Last matches
    wins=past_matches[past_matches.Winner==player]    
    losses=past_matches[past_matches.Loser==player]
    
    denom = wins.shape[0] + losses.shape[0]
    pc_win = wins.shape[0] / denom if denom > 0 else 0.5
    
    return pc_win

def FacetoFace(outcome, match, past_matches):
    player1=match.Winner if outcome==1 else match.Loser
    player2=match.Loser if outcome==1 else match.Winner
    
    duel1 = [(past_matches.Winner==player1) & (past_matches.Loser==player2)]
    duel2 = [(past_matches.Winner==player2) & (past_matches.Loser==player1)]
    
    denom = np.sum(duel1) + np.sum(duel2)
    duel = np.sum(duel1) / denom if denom >0 else 0.5 
    
    return duel
