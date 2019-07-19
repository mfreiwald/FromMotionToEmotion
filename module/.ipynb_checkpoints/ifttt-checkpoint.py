import requests

url = "https://maker.ifttt.com/trigger/Cluster_Finish/with/key/yutJVQSRvicZZZNbhS0CX?value1=%s&value2=%s"


def send(func, time):
    requests.get(url % (func, time))
