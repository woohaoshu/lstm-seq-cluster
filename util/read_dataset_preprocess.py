#coding:utf-8

import MySQLdb
import codecs
import json
import nltk
from collections import Counter


def read_api_basic(conn):
    cursor = conn.cursor()

    sql = "SELECT `ID`, `Name`, `Description` FROM `apibasic`"
    cursor.execute(sql)
    result = cursor.fetchall()

    return result

def read_api_cate(conn):
    cursor = conn.cursor()

    sql = "SELECT `ApiID`, `CateID` FROM `apicate` WHERE `isPri`=1"
    cursor.execute(sql)
    result = cursor.fetchall()

    return result


def build_dataset(api_basic, api_cate, cate_num):
    count = 0
    # 停用词表
    stopwords = []
    with codecs.open('../dataset/stopwords.txt', 'r', 'utf-8') as stop_text:
        for word in stop_text.readlines():
            stopwords.append(word.strip())

    # api与category的映射，most_common_cate拿到前cate_num个类
    api_cate_dict = dict()
    for api_id, cate_id in api_cate:
        api_cate_dict[api_id] = cate_id
    most_common_cate = Counter(api_cate_dict.values()).most_common(cate_num)
    most_common_cate_list = [x[0] for x in most_common_cate]

    # 初始化词干化
    porter_stemmer = nltk.stem.PorterStemmer() # 词根还原
    porter_lemmatize = nltk.stem.WordNetLemmatizer() # 词形还原

    # 构建api的四元集合
    desc_list = []
    for api_id, name, desc in api_basic:
        try:
            # 剔除出现频率较低的类
            if api_cate_dict[api_id] in most_common_cate_list:
                label = api_cate_dict[api_id]
            else:
                continue

            words = nltk.word_tokenize(desc) # 分词
            # filtered = [porter_stemmer.stem(w).lower() for w in words if w not in stopwords] # 去停用词、词干化并全转小写，词根还原
            filtered = [porter_lemmatize.lemmatize(w).lower() for w in words if w not in stopwords] # 词形还原
            filtered = " ".join(filtered)

            desc_list.append({
                'id': api_id,
                'name': name,
                'desc': filtered,
                'label': label
            })
            count += 1
        except:
            print("except: id:%s, name:%s" %(api_id, name))
            continue
        
    # 保存到json文件
    with codecs.open('../dataset/data_lemmatize2.json', 'w', 'utf-8') as df:
        out = json.dumps(desc_list)
        df.write(out)
    return count


if __name__ == '__main__':
    # 连接数据库
    conn = MySQLdb.connect('localhost', 'root', '', 'programmableweb_new', charset = 'utf8mb4')
    api_basic = read_api_basic(conn)
    api_cate = read_api_cate(conn)
    count = build_dataset(api_basic, api_cate, 2)
    print("The number of data is " + str(count))
