import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import altair as alt

st.markdown("# Welcome to Xchange 2023")
st.markdown("### Insights and Data in Stavanger")

### insert game token and get data
game_token = st.text_input('Game token', 'ffbe43a5-75b9-4478-a8fa-2c052ba879c2')
url = 'https://game-server.geoguessr.com/api/duels/' + game_token
cookies = {
    '_ncfa': "hpGcgybEYZQHthKKNcJau5%2BgG6TcHIqFB5NKojkbOZE%3DpunDNm4VNbjKoylctHMUwN9xgEz41gZheIssGn2A89M%2F%2FI7ZSf27qL8hO07TCyks",
    'devicetoken': '377B4D6476'
}
r = requests.get(url, cookies=cookies) 
data = json.loads(r.text)

### Normalizing data
df = pd.json_normalize(data, record_path =['teams','roundResults'], meta = [['teams','name']])
df = df.loc[:, ('teams.name','roundNumber', 'score', 'healthAfter')]
df.rename(columns = {"teams.name": "Teams", "roundNumber": "Rounds", "score": "Score", "healthAfter": "Health"}, inplace= True)

### split dataframe into 2 teams
df_red = df.loc[df['Teams'] == 'red'].loc[:, ('Score','Health')]
df_blue = df.loc[df['Teams'] == 'blue'].loc[:, ('Score','Health')]
#index_list = np.arange(1,len(df_blue)+1)
#df_red.index = index_list
#df_blue.index = index_list

### show dataframe on different tabs
st.markdown("## Game Result")
tab1, tab2, tab3 = st.tabs(["Dashboard", "Team blue and Team red", "Score Summary"])

# show game result dashboard
with tab1:
    c2 = alt.Chart(df).mark_line(point = True).encode(
    x= alt.X('Rounds:O'),
    y='Health:Q',
    color = alt.Color('Teams', scale=alt.Scale(range=['#22577A', '#CC2936'])),
    column = 'Teams:N'
    )
    st.altair_chart(c2)

    col1, col2 = st.columns(2)
    col1.markdown("- Blue team:")
    col2.markdown("- Red team:")

    col1.metric(
        label = "Score at round " + str(len(df_blue)), 
        value = df_blue['Score'].iloc[-1], 
        delta = str(df_blue['Score'].iloc[-1] - df_blue['Score'].iloc[-2]) 
        )

    col2.metric(
        label = "Score at round " + str(len(df_red)), 
        value = df_red['Score'].iloc[-1], 
        delta = str(df_red['Score'].iloc[-1] - df_red['Score'].iloc[-2])
        )

    col1.metric(
        label = "Health at round " + str(len(df_blue)), 
        value = df_blue['Health'].iloc[-1], 
        delta = str(df_blue['Health'].iloc[-1] - df_blue['Health'].iloc[-2])
    )
    col2.metric(
        label = "Health at round " + str(len(df_red)), 
        value = df_red['Health'].iloc[-1], 
        delta = str(df_red['Health'].iloc[-1] - df_red['Health'].iloc[-2])
    )

# show line chart
with tab2:
    c1 = alt.Chart(df).mark_bar().encode(
        x='Rounds:O',
        y= 'Score:Q',
        color = alt.Color('Teams:N', scale=alt.Scale(range=['#22577A', '#CC2936'])),
        column = 'Teams:N'
    )
    st.altair_chart(c1)
    col1, col2 = st.columns(2)
    col1.markdown("- Blue team:")
    col2.markdown("- Red team:")
    col1.table(df_blue)
    col2.table(df_red)

#show bar chart
with tab3:
    #col1, col2 = st.columns(2)
    c3 = alt.Chart(df).mark_bar().encode(
        x='Rounds:Q',
        y='Score:Q',
        color = alt.Color('Teams:N', scale=alt.Scale(range=['#22577A', '#CC2936'])),
    )
    c4 = alt.Chart(df).mark_bar().encode(
        x= alt.X('Teams:N',
                title = None
                ),
        y='Score:Q',
        color = alt.Color('Teams:N', scale=alt.Scale(range=['#22577A', '#CC2936'])),
        column = 'Rounds:O'
    )
    #col1.altair_chart(c3)
    #col2.altair_chart(c4)
    st.altair_chart(c4)