import os
import re
import json
import requests
import time
import urllib.parse
import random
import traceback

#定义要爬取的标签和正在爬取的页数
def UserUrl(theme,pagenum):
    url = "https://tuchong.com/rest/tags/%(theme)s/posts?page=%(pagenum)s&count=20&order=weekly" % {'theme': urllib.parse.quote(theme), 'pagenum': pagenum}
    #print(url)
    return url

#利用requests使用get方法请求url，使用User-Agent是为了防止被反爬，这样使得我们的爬取行为更像人的行为
def GetHtmltext(url):
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=head, timeout=30)
        r.raise_for_status()  #如果返回的状态码不是200，就到except中
        return r
    except:
        pass

#定义获取一个pagenum页面中的所有图集的URL链接的函数
def PictureFatherUrl(user_url):
    try:
        raw_data = GetHtmltext(user_url)
        j_raw_data = json.loads(raw_data.text)   #将获取的网页转化为Python数据结构
        # print(j_raw_data)
        father_url = []                    #将每个图集的url定义为father_url的一个列表
        for i in j_raw_data['postList']:   #解析出的j_raw_data是一个多重字典，在这里先将postList字典的内容取出来
            father_url.append(i['url'])     #然后再取出键为“url”的值
        # print(father_url)
        # print(len(father_url))
        return father_url
    except:
        return

#定义获取一个图集中所有图片的url链接
def PictureUrl(url):
    try:
        html = GetHtmltext(url)
        #利用正则表达式来匹配
        url_list = list(re.findall('<img id="image\d+" class="multi-photo-image" src="([a-zA-z]+://[^\s]*)" alt="">', html.text))
        return url_list
    except:
        pass

#定义一个图集中所有图片的下载
def Download(url):
    url_list = PictureUrl(url)
    for i in url_list:
        r = GetHtmltext(i)
        file_name = os.path.join(save_path, i.split('/')[-1])
        # file_name = str(i)
        with open(file_name, 'wb') as f:
            f.write(r.content)
            f.close()
        time.sleep(random.uniform(0.3, 0.5))  #为防止被反爬，在这里random了0.3-0.5的数，然后在下载一张图片后，sleep一下
        print('save %s' % file_name)

#定义主函数
if __name__ == '__main__':
    theme = input("你选择的标签（如果你不知道有什么标签，去https://tuchong.com/explore/去看看有什么标签吧，输入不存在的标签无法下载哦）：")
    pagenum_all = int(input("你要爬取的页数(不要太贪心哦，数字太大会被封IP的）："))
    save_path = os.path.join(theme)
    m = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("我知道你没有创建保存路径，我把文件存在和此脚本同样的路径下的叫做“ %s ”的文件夹下面了" % theme)
    for i in range(1, pagenum_all+1):
        n = 0
        m += 1
        print("正在下载第%d页，一共%d页" % (m, pagenum_all))
        user_url = UserUrl(theme, i)
        father_url = PictureFatherUrl(user_url)
        for j in father_url:
            n += 1
            print("正在下载第%d套图，一共%d套图" % (n, len(father_url)))
            Download(j)
            time.sleep(random.randint(6, 10))  #同样为了反爬，也random了6-10之间的数，更真实的模拟人的操作
