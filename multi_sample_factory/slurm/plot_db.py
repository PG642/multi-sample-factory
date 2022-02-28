from tensorflow.python.summary.summary_iterator import summary_iterator
import plotly.express as px
import sys
import os
import statistics
import plotly.graph_objects as go
import scipy.stats as st
from math import sqrt
import sqlite3

def plot_throughput():
    f = open(sys.argv[2], "r")
    lines = f.readlines()
    
    conn = sqlite3.connect('grid_search_new.db')
    cursor = conn.cursor()


    max_nodes = 0
    throughputs = []
    nodes = []
    found_new_max = True
    node_list = [1,2,4,6,8,10]
    for node in node_list:
        found_new_max = False
        throughput = 0
        found_jobs = 0
        for i in range(1, len(lines)):
            parameters = lines[i].split(";")
            if int(parameters[4]) == node:
                found_new_max = True
                #found_jobs += 1
                job_throughput = 0
                run_incomplete = False
                for j in range(node):
                    for p in range(int(parameters[1])):
                        policy_throughput = 0
                        found_runs = 0
                        for k in range(2):
                            cursor.execute("SELECT AVG (VALUE) FROM STEPS WHERE JOB=? AND NODE=? AND RERUN=? AND POLICY=? AND TAG='0_aux/_sample_throughput'", (parameters[0], j, k, p))
                            run_throughput = cursor.fetchone()[0]
                            if(run_throughput == None):
                                run_incomplete = True
                                continue
                            policy_throughput += run_throughput
                            found_runs += 1
                        if(found_runs > 0):
                            job_throughput += policy_throughput / found_runs
                if not run_incomplete:
                    found_jobs += 1
                    throughput += job_throughput
                run_incomplete = False
        if(found_new_max):
            nodes.append(node)
            throughputs.append(throughput/found_jobs)
                
    base = throughputs[0]
    throughputs = [i/base for i in throughputs]
    optimal = list(range(1,10+1))
    

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=nodes,
            y=throughputs,
            name="throughput",
            ))
    fig.add_trace(go.Scatter(
            x=nodes,
            y=nodes,
            name="optimal",
            line_color='indigo'
            ))
    fig.show()

if __name__ == '__main__':
    sys.exit(plot_throughput())
