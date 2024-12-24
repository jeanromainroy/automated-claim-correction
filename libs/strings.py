# Libraries
import re

def remove_links(text):
    return re.sub(r'http\S+|www\.\S+', '', text)
