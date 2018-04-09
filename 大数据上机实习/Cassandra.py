from cassandra.cluster import Cluster
import pandas as pd
import re

def main():
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect()
    session.execute(
        "create keyspace if not exists bigdata with replication={'class':'SimpleStrategy', 'replication_factor': 1}")
    session.execute('use bigdata')

    '''
    session.execute("create table student (id text primary key, school text, sex text, "
                    "age int, address text, famsize text, Pstatus text, Medu int, Fedu int,"
                    "Mjob text, Fjob text, reason text, guardian text, traveltime int, "
                    "studytime int, failures int, schoolsup text, famsup text, paid text, "
                    "activities text, nursery text, higher text, internet text, romantic text, "
                    "famrel int, freetime int, goout int, Dalc int, Walc int, health int, "
                    "absences int, G1 int, G2 int, G3 int)")
    '''

    data = pd.read_table('student.csv', sep=';')
    key_string = '(' + ','.join(data.columns) + ')'
    pattern = re.compile('\\[(.*)\\]')
    for i in range(len(data)):
        values_string = '(' + pattern.match(str(list(data.loc[i]))).group(1) + ')'
        insert = "insert into bigdata.student" + key_string + "values" + values_string
        session.execute(insert)

    rs = session.execute("select * from bigdata.student")
    for i in rs:
        print(i.id + ":" + str(i.age))


if __name__ == '__main__':
    main()