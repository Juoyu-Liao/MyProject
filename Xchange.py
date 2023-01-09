import streamlit as st
import requests
import json
import pandas as pd
import altair as alt

st.markdown("# Welcome to Xchange 2023")
st.markdown("### Insight and Data in Stavanger")


### insert game token and get data
game_token = st.text_input('Game token', 'c58aa4f6-ce05-4bcb-8837-fd977d2a4930')
url = 'https://game-server.geoguessr.com/api/duels/' + game_token
cookies = {
    '_ncfa': "hpGcgybEYZQHthKKNcJau5%2BgG6TcHIqFB5NKojkbOZE%3DpunDNm4VNbjKoylctHMUwN9xgEz41gZheIssGn2A89M%2F%2FI7ZSf27qL8hO07TCyks",
    'devicetoken': '377B4D6476'
}
r = requests.get(url, cookies=cookies) 
data = json.loads(r.text)

### Normalizing data
df = pd.json_normalize(data, record_path =['teams','roundResults'], meta = [['teams','name']])
df.drop(columns = ['bestGuess.roundNumber', 'bestGuess.lat', 'bestGuess.lng', 'bestGuess.distance', 'bestGuess.created', 'bestGuess.isTeamsBestGuessOnRound'], inplace = True)
df.rename(columns = {"teams.name": "teams"}, inplace= True)
### split dataframe into 2 teams
df_red = df.loc[df['teams'] == 'red']
df_red.set_index(['roundNumber'], inplace = True)
df_red.drop(columns = ['healthBefore', 'teams'], inplace = True)
df_blue = df.loc[df['teams'] == 'blue']
df_blue.set_index(['roundNumber'], inplace = True)
df_blue.drop(columns = ['healthBefore', 'teams'], inplace = True)


### show dataframe on different tabs
st.markdown("## Game Result")
tab1, tab2, tab3 = st.tabs(["Dashborad", "Team blue and Team red", "Summary"])

# show game result dashboard
with tab1:
    col1, col2 = st.columns(2)
    col1.markdown("- Blue team:")
    col2.markdown("- Red team:")

    col1, col2 = st.columns(2)
    col1.metric(
        label = "Blue team score at round number " + str(len(df_blue)), 
        value = df_blue['score'].loc[len(df_blue)], 
        delta = str(df_blue['score'].loc[len(df_blue)] - df_blue['score'].loc[len(df_blue)-1]) 
        )

    col2.metric(
        label = "Red team score at round number " + str(len(df_red)), 
        value = df_red['score'].loc[len(df_red)], 
        delta = str(df_red['score'].loc[len(df_red)] - df_red['score'].loc[len(df_red)-1])
        )

    col1, col2 = st.columns(2)
    col1.metric(label = "Blue team health at round number " + str(len(df_blue)), 
    value = df_blue['healthAfter'].loc[len(df_blue)], 
    delta = str(df_blue['healthAfter'].loc[len(df_blue)] - df_blue['healthAfter'].loc[len(df_blue)-1])
    )
    col2.metric(label = "Red team health at round number " + str(len(df_red)), 
    value = df_red['healthAfter'].loc[len(df_red)], 
    delta = str(df_red['healthAfter'].loc[len(df_red)] - df_red['healthAfter'].loc[len(df_red)-1])
    )

    col1, col2 = st.columns(2)
    col1.table(df_blue)
    col2.table(df_red)

# show line chart
with tab2:
    col1, col2 = st.columns(2)
    c1 = alt.Chart(df).mark_line(point = True).encode(
        x=alt.X('roundNumber:O', title= None),
        y='score:Q',
        color = 'teams:N',
        column = 'teams:N'
    )
    c2 = alt.Chart(df).mark_line(point = True).encode(
        x= alt.X('roundNumber:O', title =None),
        y='healthAfter:Q',
        color = 'teams:N',
        column = 'teams:N'
    )
    col1.altair_chart(c1)
    col2.altair_chart(c2)

#show bar chart
with tab3:
    c = alt.Chart(df).mark_bar().encode(
        x='roundNumber:Q',
        y='score:Q',
        color = 'teams:N',
    )
    st.altair_chart(c)
    
    c = alt.Chart(df).mark_bar().encode(
        x='teams:N',
        y='score:Q',
        color = 'teams:N',
        column = 'roundNumber:O'
    )
    st.altair_chart(c)

    text = c.mark_text(
        dx = 3
    ).encode(
        text = 'score:Q'
    )
    (c + text).properties(height=900)