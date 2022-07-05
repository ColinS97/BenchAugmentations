import plotly.express as px
import pandas as pd

none_df = pd.read_csv("plot/pathmnist_none_100_220626_193330_Tensorboard_Results.csv")
none_df.drop(columns=["Wall time"], inplace=True)

none_df.rename(columns={"Value": "None"}, inplace=True)

trivialaugment_df = pd.read_csv(
    "plot/pathmnist_trivialaugment_100_220701_090201_Tensorboard_Results.csv"
)
trivialaugment_df.drop(columns=["Wall time"], inplace=True)

trivialaugment_df.rename(columns={"Value": "Trivialaugment"}, inplace=True)

randaugment_df = pd.read_csv(
    "plot/pathmnist_randaugment_100_220701_104540_Tensorboard_Results.csv"
)
randaugment_df.drop(columns=["Wall time"], inplace=True)

randaugment_df.rename(columns={"Value": "Randaugment"}, inplace=True)

# join the dataframes on step
none_df = none_df.merge(trivialaugment_df, on="Step")
new_df = none_df.merge(randaugment_df, on="Step")
# none_df - none_df.merge(, on="Step")
print(new_df.head())
fig = px.line(
    new_df,
    x="Step",
    y=["None", "Trivialaugment", "Randaugment"],
    labels={"x": "Epoch", "y": "Accuracy"},
    title="Accuracy of Models",
)
fig.data[0].line.color = "#f826d6"

fig.data[1].line.color = "#dbdbdb"

fig.data[2].line.color = "#46e7f8"

fig.write_image("fig_hd.png", width=1920, height=1080)
