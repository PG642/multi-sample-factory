from tensorflow.python.summary.summary_iterator import summary_iterator
import plotly.express as px
import sys
import os
import statistics
import plotly.graph_objects as go

file_list = []
for dirpath,_,files in os.walk(sys.argv[1]):
    for f in files:
        if f.startswith("events.out.tfevents"):
            file_list.append(os.path.join(dirpath, f))
            

summary_file = file_list[0]
tags = []
for e in summary_iterator(file_list[0]):
    for v in e.summary.value:
        if v.tag not in tags:
            tags.append(v.tag)

tags.sort()


values = [None]* len(file_list)
for i in range(len(values)):
    values[i] = [None] * len(tags)
for i in range(len(values)):
    for j in range(len(values[i])):
        values[i][j] = [[], []]
    
for i in range(len(values)):
    for e in summary_iterator(file_list[i]):
        for v in e.summary.value:
            values[i][tags.index(v.tag)][0].append(e.step)
            values[i][tags.index(v.tag)][1].append(v.simple_value)


for i in range(len(tags)):
    print(f"{i+1}: {tags[i]}")
    
x = int(input("Select tag: "))

value_lists = []
for i in range(len(values)):
    value_lists.append(values[i][x-1][1])

mean_list = list(map(statistics.mean, zip(*value_lists)))
min_list = list(map(min, zip(*value_lists)))
max_list = list(map(max, zip(*value_lists)))

diff_min = list(map(float.__sub__, mean_list, min_list))
diff_max = list(map(float.__sub__, max_list, mean_list))


#draw figure
#fig = px.line(x=values[x-1][0], y=values[x-1][1], title='Test')
#fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(
        x=values[0][x-1][0],
        y=min_list,
        name="min",
        line_color='indigo'
        ))
fig.add_trace(go.Scatter(
        x=values[0][x-1][0],
        y=max_list,
        name="max",
        fill="tonexty",
        line_color='indigo'
        ))
fig.add_trace(go.Scatter(
        x=values[0][x-1][0],
        y=mean_list,
        name="mean",
        ))
fig.show()
