import streamlit as st
import requests
import json
import pandas as pd
import altair as alt

st.markdown("# Welcome to Xchange 2023")
st.markdown("### Insight and Data in Stavanger")

### insert game token and get data
game_token = st.text_input('Game token', '817f5ea9-9651-4190-be01-a8d7e852af61')
url = 'https://game-server.geoguessr.com/api/duels/' + game_token
cookies = {
    '_ncfa': "hpGcgybEYZQHthKKNcJau5%2BgG6TcHIqFB5NKojkbOZE%3DpunDNm4VNbjKoylctHMUwN9xgEz41gZheIssGn2A89M%2F%2FI7ZSf27qL8hO07TCyks",
    'devicetoken': '377B4D6476'
}
r = requests.get(url, cookies=cookies) 
data = json.loads(r.text)

### Normalizing data
df = pd.json_normalize(data, record_path =['teams','roundResults'], meta = [['teams','name']])
df.drop(columns = ['bestGuess.roundNumber', 'bestGuess.lat', 'bestGuess.lng', 'bestGuess.distance', 'bestGuess.created', 'bestGuess.isTeamsBestGuessOnRound'], 
        inplace = True
        )
df.rename(columns = {"teams.name": "Teams", "roundNumber": "Rounds", "score": "Score", "healthAfter": "Health"}, inplace= True)

### split dataframe into 2 teams
#df_red = df.loc[df['Teams'] == 'red'].loc[:, ('Score','Health')]
#df_blue = df.loc[df['Teams'] == 'blue'].loc[:, ('Score','Health')]
df_red = df.loc[df['Teams'] == 'red']
df_red.set_index(['Rounds'], inplace = True)
df_red.drop(columns = ['healthBefore', 'Teams'], inplace = True)
df_blue = df.loc[df['Teams'] == 'blue']
df_blue.set_index(['Rounds'], inplace = True)
df_blue.drop(columns = ['healthBefore', 'Teams'], inplace = True)

### show dataframe on different tabs
st.markdown("## Game Result")
tab1, tab2, tab3 = st.tabs(["Dashboard", "Team blue and Team red", "Score Summary"])

# show game result dashboard
with tab1:
    c2 = alt.Chart(df).mark_line(point = True).encode(
    x= alt.X('Rounds:O'),
    y='Health:Q',
    color = 'Teams:N',
    column = 'Teams:N'
    )
    st.altair_chart(c2)

    col1, col2 = st.columns(2)
    col1.markdown("- Blue team:")
    col2.markdown("- Red team:")

    col1.metric(
        label = "Score at round " + str(len(df_blue)), 
        value = df_blue['Score'].loc[len(df_blue)], 
        delta = str(df_blue['Score'].loc[len(df_blue)] - df_blue['Score'].loc[len(df_blue)-1]) 
        )

    col2.metric(
        label = "Score at round " + str(len(df_red)), 
        value = df_red['Score'].loc[len(df_red)], 
        delta = str(df_red['Score'].loc[len(df_red)] - df_red['Score'].loc[len(df_red)-1])
        )

    # col1.metric(
    #     label = "Health at round " + str(len(df_blue)), 
    #     value = df_blue['Health'].loc[len(df_blue)], 
    #     delta = str(df_blue['Health'].loc[len(df_blue)] - df_blue['Health'].loc[len(df_blue)-1])
    # )
    # col2.metric(
    #     label = "Health at round " + str(len(df_red)), 
    #     value = df_red['Health'].loc[len(df_red)], 
    #     delta = str(df_red['Health'].loc[len(df_red)] - df_red['Health'].loc[len(df_red)-1])
    # )

# show line chart
with tab2:
    c1 = alt.Chart(df).mark_bar().encode(
        x='Rounds:O',
        y= 'Score:Q',
        color = 'Teams:N',
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
        color = 'Teams:N',
    )
    c4 = alt.Chart(df).mark_bar().encode(
        x= alt.X('Teams:N',
                title = None
                ),
        y='Score:Q',
        color = 'Teams:N',
        column = 'Rounds:O'
    )
    #col1.altair_chart(c3)
    #col2.altair_chart(c4)
    st.altair_chart(c4)