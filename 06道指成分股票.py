# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:44:21 2020

@author: 宋海沁
"""

import json
import re
import requests

def getHtmlText():
    try:
        r = requests.get("http://money.cnn.com/data/dow30/")
    except Exception as e:
        print(e)
    search_pattern=re.compile('class="wsod-symbol">(.*?)<\/a>.*<span.*>\(.*?)<\/span>.*\n.*class="wsod-stream">(.*?)<\/span>")
    dj_list_in_text = re.findall(search_patterb,r.text)
    
    