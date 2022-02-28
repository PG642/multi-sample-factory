from tensorflow.python.summary.summary_iterator import summary_iterator
import sys
import os
import sqlite3

f = open(sys.argv[2], "r")
lines = f.readlines()

conn = sqlite3.connect('grid_search_new.db')

conn.execute('''CREATE TABLE STEPS
         (ID            INT  NOT NULL,
         STEP           INT  NOT NULL,
         VALUE          REAL NOT NULL,
         JOB            TEXT NOT NULL,
         NODE           INT  NOT NULL,
         RERUN          INT  NOT NULL,
         POLICY         INT  NOT NULL,
         BATCH_SIZE     INT  NOT NULL,
         NUM_WORKERS    INT  NOT NULL,
         TAG            TEXT NOT NULL);''')
conn.commit()

conn.execute('''CREATE INDEX SORT_INDEX
                ON STEPS (JOB, NODE, RERUN, POLICY, TAG, STEP);''')
conn.commit()


cursor = conn.cursor()
# id, step, value, job, node, rerun, policy, batch_size, num_envs_per_worker, tag

id=0
for line in range(1, len(lines)):
    parameters = lines[line].split(";")
    job_name = parameters[0]
    name = job_name.rsplit("_", 1)[0]
    run = int(job_name.rsplit("_", 1)[1])
    num_policies = int(parameters[1])
    batch_size = int(parameters[2])
    num_envs_per_worker = int(parameters[3])
    num_nodes = int(parameters[4])
    num_reruns = 3
    for i in range(num_nodes):
        for j in range(num_reruns):
            for k in range(num_policies):
                path = sys.argv[1] + name + "_" + str(i) + "/" + name + "_" + f"{run:03d}" + "_" + f"{j:03d}" + "_" + str(i) + "/" + ".summary" + "/" + str(k) + "/"
                try:
                    file_path = os.path.join(path, os.listdir(path)[0])
                
                except FileNotFoundError as e:
                    print(e)
                    continue
                for e in summary_iterator(file_path):
                    for v in e.summary.value:
                        cursor.execute("INSERT INTO STEPS (ID,STEP,VALUE,JOB,NODE,RERUN,POLICY,BATCH_SIZE,NUM_WORKERS,TAG) \
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (id, e.step, v.simple_value, job_name, i, j, k, batch_size, num_envs_per_worker, v.tag));
                        id += 1

conn.commit()
conn.close()
