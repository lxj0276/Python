# encoding:utf-8

import requests
from prettytable import PrettyTable
from colorama import Fore,Style


def head(s):
    return Fore.RED + Style.BRIGHT + s + Fore.WHITE  # 调回白色否则会影响之后的输出


def name(s):
    return Fore.BLUE + s + Fore.WHITE


def like(n):
    s = '%d' % n
    return Fore.RED + s + Fore.WHITE


def comment(n):
    s = '%d' % n
    return Fore.CYAN + Style.DIM + s + Fore.WHITE


h = {
    'X-REQUESTID': '585285862',
    'User-Agent': 'kwai-android',
    'Connection': 'keep-alive',
    'Accept-Language': 'zh-cn',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Content-Length': '154',
    'Host': '114.118.4.4',
    'Accept-Encoding': 'gzip'
}

url = ('http://114.118.4.4/rest/n/feed/hot?mod=HUAWEI(HUAWEI%20MLA-AL10)&lon=0&country_code=cn'
       '&did=ANDROID_7932d26f7c57ad1e&net=WIFI&app=0&oc=HUAWEI&ud=0&c=HUAWEI&sys=ANDROID_7.0'
       '&appver=5.6.0.5887&ftt=&language=zh-cn&iuid=&lat=0&ver=5.6&max_memory=384')
postData = {
    "type": "7",
    "page": "3",
    "coldStart": "false",
    "count": "20",
    "pv": "false",
    "id": "3",
    "refreshTimes": "2",
    "pcursor": "1",
    "source": "1",
    "client_key": "3c2cd3f3",
    "os": "android",
    "sig": "9666cfa6250f4710e8a0f974c4803ba2"
}

with open('result.txt', 'w', encoding='utf-8') as fp:
    stopTag = 0
    tableHead = ['用户', '点赞', '评论', '播放数']
    table = PrettyTable([head(i) for i in tableHead])
    while(stopTag < 100):
        response = requests.post(url, data=postData)
        result = response.json()
        if len(result) != 2 :
            for i in result['feeds']:
                line = [name(i['user_name']), like(i['like_count']), i['comment_count'], i['view_count']]
                table.add_row(line)
                fp.write('{user_name} {like_count} {comment_count} {view_count}'.format(**i))
                fp.write('\n')
        else:
            print(result)
        stopTag += len(result['feeds'])
    print(table)