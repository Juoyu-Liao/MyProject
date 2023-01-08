import streamlit as st
import requests
import pandas as pd
import json


game_token = '3f325168-e2b8-4957-9533-5297fa7b999f'  #game token needed to be chagned 

url = 'https://game-server.geoguessr.com/api/duels/' + game_token
cookies = {
    '_ncfa': "hpGcgybEYZQHthKKNcJau5%2BgG6TcHIqFB5NKojkbOZE%3DpunDNm4VNbjKoylctHMUwN9xgEz41gZheIssGn2A89M%2F%2FI7ZSf27qL8hO07TCyks",
    'devicetoken': '377B4D6476'
}
r = requests.get(url, cookies=cookies) 
data = json.loads(r.text)

# Normalizing data
df = pd.json_normalize(data, record_path =['teams','roundResults'], meta = [['teams','name']])
df.drop(columns = ['bestGuess.roundNumber', 'bestGuess.lat', 'bestGuess.lng', 'bestGuess.distance', 'bestGuess.created', 'bestGuess.isTeamsBestGuessOnRound'], inplace = True)
df.rename(columns = {"teams.name": "teams"}, inplace= True)

# df.set_index('roundNumber', inplace = True)
df_red = df.loc[df['teams'] == 'red']
df_red.set_index(['roundNumber'], inplace = True)
df_red.drop(columns = ['healthBefore', 'teams'], inplace = True)
df_blue = df.loc[df['teams'] == 'blue']
df_blue.set_index(['roundNumber'], inplace = True)
df_blue.drop(columns = ['healthBefore', 'teams'], inplace = True)

#streamlit
st.markdown("# Welcome to Xchange")

# show dataframe
st.markdown("## Game Result")
col1, col2 = st.columns(2)
col1.metric(label = "Red team score at round number " + str(len(df_red)), value = df_red['score'].loc[len(df_red)], delta = str(df_red['score'].loc[len(df_red)] - df_red['score'].loc[len(df_red)-1]) )
col2.metric(label = "Blue team score at round number " + str(len(df_blue)), value = df_blue['score'].loc[len(df_blue)], delta = str(df_blue['score'].loc[len(df_blue)] - df_blue['score'].loc[len(df_blue)-1]) )

col1, col2 = st.columns(2)
col1.metric(label = "Red team health at round number " + str(len(df_red)), value = df_red['healthAfter'].loc[len(df_red)], delta = str(df_red['healthAfter'].loc[len(df_red)] - df_red['healthAfter'].loc[len(df_red)-1]))
col2.metric(label = "Blue team health at round number " + str(len(df_blue)), value = df_blue['healthAfter'].loc[len(df_blue)], delta = str(df_blue['healthAfter'].loc[len(df_blue)] - df_blue['healthAfter'].loc[len(df_blue)-1]))

col1, col2 = st.columns(2)
col1.markdown("- Red team:")
col2.markdown("- Blue team:")
col1, col2 = st.columns(2)
col1.table(df_red)
col2.table(df_blue)

#show bar chart
st.markdown("## Team red and Team blue")

import altair as alt
c = alt.Chart(df).mark_bar().encode(
    x='roundNumber:Q',
    y='score:Q',
    color = 'teams:N',
    #column = 'roundNumber'
)
st.altair_chart(c)

col1, col2 = st.columns(2)
c1 = alt.Chart(df).mark_bar().encode(
    x='roundNumber:O',
    y='score:Q',
    color = 'teams:N',
    column = 'teams:N'
)
c2 = alt.Chart(df).mark_bar().encode(
    x='roundNumber:O',
    y='healthAfter:Q',
    color = 'teams:N',
    column = 'teams:N'
)

col1.altair_chart(c1)
col2.altair_chart(c2)

c = alt.Chart(df).mark_bar().encode(
    x='teams:N',
    y='score:Q',
    color = 'teams:N',
    column = 'roundNumber:O'
)
st.altair_chart(c)