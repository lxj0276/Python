# encoding:utf-8
from pymongo import MongoClient
from collections import defaultdict
import redis
from cassandra.cluster import Cluster
import time


def test_dict():
    l = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    d = defaultdict(int)
    for i in l:
        d[i] += 1
    print(sorted(d.keys(), key=lambda obj:d[obj], reverse=True)[0])


def make_dict(names, values):
    return {names[i]: values[i] for i in range(len(names))}


def test_file(n):
    with open('yagoThreeSimplified.txt','r') as f:
        with open('test.txt','w') as fw:
            for i in range(n):
                fw.write(f.readline())


def mdb_store(file, names, db, sep):
    with open(file) as f:
        for line in f:
            line = line.strip('\n')
            db.insert_one(make_dict(names, line.split(sep)))


def mdb_main(file, sep, arg_list, store=False):
    # 连接MongoDB服务
    client = MongoClient('localhost', 27017)
    # 选择数据库及集合
    db = client.test
    lord = db.lord
    names = ['S', 'P', 'O']

    if store:
        mdb_store(file, names, lord, sep)
    # 存入数据库

    # 有S，查所有P，O
    S = arg_list[0]
    rs = lord.find({'S': S})
    for tmp in rs:
        print((tmp['P'], tmp['O']))

    # 有O，查S，P
    O = arg_list[1]
    rs = lord.find({'O': O})
    for tmp in rs:
        print((tmp['P'], tmp['O']))

    # 给出p1，p2，查同时拥有它们的S
    p1, p2 = arg_list[2], arg_list[3]
    rs1 = lord.find({'P': p1})
    rs2 = lord.find({'P': p2})
    result = set([i['S'] for i in rs1]).intersection(set([i['S'] for i in rs2]))
    print(result)

    # 给定O，查拥有O最多的S
    O = arg_list[4]
    rs = lord.find({'O': O})
    d = defaultdict(int)
    for tmp in rs:
        d[tmp['S']] += 1
    print(sorted(d.keys(), key=lambda obj:d[obj], reverse=True)[0] if len(d) > 0 else None)


def redis_store(file, r, sep):
    with open(file, 'r') as f:
        for line in f:
            try:
                line = line.strip('\n')
                s, p, o = line.split(' ')
                r.sadd(s, '{} {}'.format(p, o))
                r.sadd(o, '{} {}'.format(s, p))
                r.sadd(p, s)
                r.hsetnx(o + 'hash', s, 0)  # key-hash表，键值key不能重复
                r.hincrby(o + 'hash', s, 1)  # 递增计数
            except:
                pass



def redis_main(file, sep, arg_list, store=False):
    # 需要注意键值不能重复
    # 连接redis
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    if store:
        redis_store(file, r, sep)

    # 有S，查所有P，O
    rs = r.smembers(arg_list[0])
    print('有S，查所有P，O：')
    for i in rs:
        print(i)

    # 有O，查S，P
    rs = r.smembers(arg_list[1])
    print('有O，查S，P：')
    for i in rs:
        print(i)

    # 给出p1，p2，查同时拥有它们的S
    p1, p2 = arg_list[2], arg_list[3]
    rs = r.sinter(p1, p2)
    print('给出p1，p2，查同时拥有它们的S：')
    for i in rs:
        print(i)

    # 给定O，查拥有O最多的S
    rs = r.hgetall(arg_list[4]+'hash')
    print('给定O，查拥有O最多的S：')
    result = sorted(rs.keys(), key=lambda obj:rs[obj], reverse=True)[0] if len(rs)>0 else None
    print(result)
    print(rs.get(result, None))
    print(rs[b'Liam_Tyson'])


def cassandra_store(file, session, sep):
    key_string = '(id,S1,P1,O1)'
    id = 0
    with open(file) as f:
        for line in f:
            line = line.strip('\n')
            id += 1
            s, p, o = line.split(' ')
            values_string = "({},'{}','{}','{}')".format(id,s,p,o)
            insert = "insert into bigdata.lord" + key_string + "values" + values_string
            session.execute(insert)


def cassandra_main(file, sep, arg_list, store=False):
    # 需要注意查询插入语句中字符串引号的使用
    # allow data filter
    # 键不区分大小写
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect()
    session.execute(
        "create keyspace if not exists bigdata with replication={'class':'SimpleStrategy', 'replication_factor': 1}")
    session.execute('use bigdata')
    session.execute('create table lord')
    session.execute('create table lord (id int primary key, S1 text, P1 text, O1 text)')

    if store:
        cassandra_store(file, session, sep)

    # 有S，查所有P，O
    rs = session.execute("select * from bigdata.lord where S1='{}' allow filtering".format(arg_list[0]))
    print('查询1：')
    for i in rs:
        print('{},{}'.format(i.s1, i.o1))

    # 有O，查S，P
    rs = session.execute("select * from bigdata.lord where O1='{}' allow filtering".format(arg_list[1]))
    print('查询2：')
    for i in rs:
        print('{},{}'.format(i.s1, i.p1))

    # 给出p1，p2，查同时拥有它们的S
    rs1 = session.execute("select * from bigdata.lord where P1='{}' allow filtering".format(arg_list[2]))
    rs2 = session.execute("select * from bigdata.lord where P1='{}' allow filtering".format(arg_list[3]))
    result_set = set(rs1).intersection(set(rs2))
    print('查询3：')
    for i in result_set:
        print(i.s1)

    # 给定O，查拥有O最多的S
    rs = session.execute("select * from bigdata.lord where O1='{}' allow filtering".format(arg_list[4]))
    d = defaultdict(int)
    for i in rs:
        d[i.s1] += 1
    result = sorted(d.keys(), key=lambda obj:d[obj], reverse=True)[0] if len(d)>0 else None
    print('查拥有 {} 最多的S：{},拥有:{}'.format(arg_list[4], result, d.get(result, None)))


def main():
    S1 = 'Liam_Tyson'
    O1 = 'France'
    P1, P2 = 'isCitizenOf', 'owns'
    O2 = 'Cuba'
    arg_list = [S1, O1, P1, P2, O2]

    start = time.time()
    # mdb_main('yagoThreeSimplified.txt', ' ', arg_list, True)
    redis_main('yagoThreeSimplified.txt', ' ', arg_list, True)
    # cassandra_main('yagoThreeSimplified.txt', ' ', arg_list, True)
    end = time.time()
    print('total cost:', end - start)


if __name__ == '__main__':
    main()

