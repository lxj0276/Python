# encoding:utf-8
import redis
import pandas as pd


def main():
    student = pd.read_table('student.csv', sep=';')
    names = list(student.columns)
    
    # 连接redis
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    # 查看服务运行状态
    print(r.ping())

    for i in range(len(names)):
        r.lpush(names[i], list(student.iloc[:,i]))

    l = r.lrange("id", 0, 2);
    for i in l:
        print(i)


if __name__ == '__main__':
    main()