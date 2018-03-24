# encoding:utf-8
import requests

dic = {}
with open('cityCode.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        pair = line.split(' ')
        dic[pair[0]] = pair[1][:3]
# cityCode.txt保存从另一个网站上获取的机场地区码

h = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': '_abtest_userid=f2cc6aa2-10ea-4c94-b516-6c0f8e5d5afa; _RSG=Llad7XL06xCGYtzQ5hZHLB; _RDG=28a186b48db35b21f0252d6e064310b6b7; _RGUID=cec60158-479e-4bd5-b551-8eb15af77890; _ga=GA1.2.1459227212.1518525743; AHeadUserInfo=VipGrade=0&UserName=&NoReadMessageCount=0&U=DEB85960291926EAE5C2A3853C61FE35; login_uid=38229535701013147033DEC550785721; login_type=6; login_cardType=6; _fpacid=09031121210319397962; GUID=09031121210319397962; Union=SID=155952&AllianceID=4897&OUID=baidu81|index|||; Session=SmartLinkCode=U155952&SmartLinkKeyWord=&SmartLinkQuary=&SmartLinkHost=&SmartLinkLanguage=zh; adscityen=Beijing; DomesticUserHostCity=BJS|%b1%b1%be%a9; _RF1=111.205.230.123; _gid=GA1.2.1010215998.1521168349; MKT_Pagesource=PC; appFloatCnt=1; FD_SearchHistorty={"type":"S","data":"S%24%u5317%u4EAC%28BJS%29%24BJS%242018-04-30%24%u5357%u4EAC%28NKG%29%24NKG"}; _bfa=1.1514200956571.2mlz2b.1.1521168340223.1521178764193.4.39.10320673802; _bfs=1.3; Mkt_UnionRecord=%5B%7B%22aid%22%3A%224897%22%2C%22timestamp%22%3A1521179672932%7D%5D; _jzqco=%7C%7C%7C%7C1521168349421%7C1.298181980.1518525743457.1521179659850.1521179672954.1521179659850.1521179672954.undefined.0.0.10.10; __zpspc=9.4.1521178767.1521179672.3%231%7Cbaiduppc%7Cbaidu%7Cty%7C%25E6%259C%25BA%25E7%25A5%25A8%7C%23; _bfi=p1%3D101027%26p2%3D101027%26v1%3D39%26v2%3D38',
    'Host': 'flights.ctrip.com',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla / 5.0(Windows NT 6.3;Win64;x64) AppleWebKit/537.36(KHTML, likeGecko) Chrome/65.0.3325.162 Safari/537.36'
   }

str = input('出发地，目的地，日期：')
params = str.split('，')
url = 'http://flights.ctrip.com/domesticsearch/search/SearchFirstRouteFlights?DCity1='\
      + dic[params[0]] + '&ACity1=' + dic[params[1]] + '&SearchType=S&DDate1=' + params[2]\
      + '&IsNearAirportRecommond=0&LogToken=eb814aad67ae464ebefca3dfee07e973&rk=3.8057681430740264135748&CK=9CDBFB9FAB8301EEE179DD539F3DF697&r=0.51167622662471653031418'

req = requests.get(url, headers = h)

result = req.json()
for i in result['fis']:
    print('航班号：{fn} 出发时间：{dt} 到达时间：{at} \n 出发机场：{dpbn} 到达机场：{apbn} \n 价格：{lp}'.format(**i))

