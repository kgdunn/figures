# Generates raincloud-for-two-by-six-100-boards.png for the data
# visualization chapter, using the same raincloud() call shown in the
# chapter's code block. Requires: pip install process_improve[plotting] kaleido
import pandas as pd
from process_improve.visualization import raincloud

all_boards = pd.read_csv("http://openmv.net/file/six-point-board-thickness.csv")
boards = all_boards.iloc[0:100, 1:7]

stacked = boards.melt(var_name="Position", value_name="Thickness")
fig = raincloud(stacked, value="Thickness", group="Position")
fig.update_layout(
    xaxis_title_text="Thickness [mils]",
    width=900,
    height=500,
    margin=dict(l=60, r=20, t=20, b=60),
)
fig.write_image("raincloud-for-two-by-six-100-boards.png", scale=2)
