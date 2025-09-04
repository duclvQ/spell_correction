
TXT_PATH = "title_list.txt"

def load_titles():
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def is_matched_titleName(text, name):
    titles = load_titles()
    text = text.lower()
    name = name.lower()
    segmented_text_list = text.split(" ")
    for title in titles:
    
        title = title.lower()
        if title in text or title.replace("_", " ") in text:
            return True
    return False