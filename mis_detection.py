TXT = "Trung tâm Dự báo Khí tượng Thủy văn quốc gia cho biết lúc 7h hôm na, áp thấp nhiệt đới mạnh 61 km/h, cấp 6-7,"
# replace uỷ with ủy
dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }
from name_checker import check_and_correct_word
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


ner_model_path = "NlpHUST/ner-vietnamese-electra-base"
def get_raw_ner(model_path:str, query: str):
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")
    model = AutoModelForTokenClassification.from_pretrained(f"{model_path}")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(query)
    return ner_results
def extract_entities(ner_results):
    entities = []
    current_entity = ""
    current_type = None


    for item in ner_results:
        ent_type = item['entity'].split('-')[-1]
        if item['entity'].startswith('B-'):
            if current_entity:
                entities.append((current_entity.strip(), current_type))
            current_entity = item['word']
            current_type = ent_type
        elif item['entity'].startswith('I-') and current_type == ent_type:
            current_entity += " " + item['word']
        else:
            if current_entity:
                entities.append((current_entity.strip(), current_type))
            current_entity = ""
            current_type = None

    if current_entity:
        entities.append((current_entity.strip(), current_type))

    return entities


from pyvi import ViTokenizer, ViPosTagger
import time
ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội")
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
spell_pipeline = pipeline(
    task="fill-mask",
    model="vinai/phobert-base",
    torch_dtype=torch.float16,
    device=0
)
phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
def process_text(text):
    for key, value in dict_map.items():
        text = text.replace(key, value)
    return text
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def sentence_prediction(text, top_k=30):
    ner_results = get_raw_ner(ner_model_path, text)

    
    entities = extract_entities(ner_results)
    
    # get entities
    print(f"Entities: {entities}")
    # create a dict the replace entities with an alias
    entity_dict = {}
    for idx, entity in enumerate(entities):
        alias = f"người_{idx}"
        entity_dict[alias] = entity[0]
        text = text.replace(entity[0], alias)
    print(f"Entity Dict: {entity_dict}" )
    text = ViTokenizer.tokenize(text)
    error_indices = []
    for i in range(len(text.split())):
        text = process_text(text)
        new_list = text.split()
        original_word = new_list[i]
        if is_number(original_word):
            continue
        # if in entities, skip
        if original_word in entity_dict.keys():
            oriname = entity_dict[original_word]
            # add _ to the word
            oriname = oriname.replace(" ", "_").lower()
            word, is_correct, suggestion = check_and_correct_word(oriname)
            if is_correct:
                continue
            else:
                error_indices.append((i, oriname, suggestion))
            print(f"Original Word: {original_word}")
            continue
        if "_" in original_word:
            continue
        new_list[i] = tokenizer.mask_token
        masked_text = " ".join(new_list)
        start_time = time.time()
        predictions = spell_pipeline(masked_text, top_k=100)
        topk: list[Unknown] = []
        for i in range(len(predictions)):
            # lower the prediction
            predictions[i]['token_str'] = predictions[i]['token_str'].lower()
        for item in predictions:
            if 'token_str' in item:
                topk.append(item['token_str'])
        end_time = time.time()
        if original_word.lower() not in topk:
            print(f"Original Word: {original_word}")
            print(f"Masked Text: {masked_text}")
            print(f"->Top 10 Predictions: {topk}")
            error_indices.append((i, original_word, topk[0]))
        # print(f"Masked Text: {masked_text}")
    return error_indices





