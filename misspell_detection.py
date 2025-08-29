from transformers.models.canine.tokenization_canine import MASK
import numpy as np
import time 
import copy
from datetime import datetime
import os
from seq2seq import seq_2seq_correct_spelling
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def is_datetime(s: str, fmt_list=None) -> bool:
    # nếu không truyền format thì dùng default phổ biến
    if fmt_list is None:
        fmt_list = [
            "%d/%m/%Y",  # 26/07/2020
            "%d-%m-%Y",  # 26-07-2020
            "%Y-%m-%d",  # 2020-07-26
            "%m/%d/%Y",  # 07/26/2020
            "%d/%m/%y",  # 26/07/20
        ]
    for fmt in fmt_list:
        try:
            datetime.strptime(s, fmt)
            return True
        except ValueError:
            continue
    return False
def is_comma_inside_number(s: str) -> bool:
    """Check if a string is a number with comma inside"""
    if "," not in s: 
        return False
    s = s.replace(",", "")
    try:
        float(s)
        return True
    except ValueError:
        return False
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
removed_punctuation = [
    ",", "!", "?", ".", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*",
    "+", "-", "=", "~", "`"
]

def remove_punctuation(text):
    """Remove punctuation from the text"""
    for punct in removed_punctuation:
        text = text.replace(punct, "")
    return text
import re
import difflib
def split_into_sentences(text):
    # split by . and add . back
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences
def replace_underscore_outside_tags(text):
    # tìm tất cả phần <...> và giữ nguyên, replace ngoài nó
    parts = re.split(r'(<[^>]+>)', text)  # split nhưng vẫn giữ <...>
    for i, p in enumerate(parts):
        if not (p.startswith("<") and p.endswith(">")):
            parts[i] = p.replace("_", " ")  # chỉ replace ngoài tag
    return "".join(parts)
def is_inside(pos1, pos2):
    """Check if pos1 is inside pos2"""
    return pos1[0] >= pos2[0] and pos1[1] <= pos2[1]
def clean_sentence(text):
    if not text:
        return text
    
    text = re.sub(r'\s+', ' ', text)                         # bỏ space thừa
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)             # bỏ space trước dấu câu
    text = re.sub(r'(?<!\d)([.,!?;:])(?!\s|$)', r'\1 ', text)
    text = re.sub(r'\s+([\)\]])', r'\1', text)               # bỏ space trước ) ]
    text = re.sub(r'([\)\]])(?![\s.,!?;:])', r'\1 ', text)   # thêm space sau ) ]
    text = re.sub(r'\.{2,}', '.', text)                      # nhiều dấu chấm -> 1
    text = text[0].upper() + text[1:] if text else text      # viết hoa đầu câu
    # after punctuation must be a space and capitalized
    text = re.sub(r'([.!?:])\s*(\w)', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
    # if not re.search(r'[.?!]$', text):                       # kết thúc bằng dấu câu
    #     text += '.'
    # text = text.replace("chào", "_")
    return text.strip()
def custom_tokenize(s):
    # match marker >"<...>" hoặc các từ thường
    return re.findall(r'>"<[^"]+>"|[^\s]+', s)

# def get_replacement_from_formatted_sentence(text, cleaned_text):
#     text_tokens = custom_tokenize(text)
#     cleaned_tokens = custom_tokenize(cleaned_text)
#     replacements = []
    
#     matcher = difflib.SequenceMatcher(None, text_tokens, cleaned_tokens)
#     for tag, i1, i2, j1, j2 in matcher.get_opcodes():
#         if tag != "equal":
#             replacements.append({
#                 "op": tag, 
#                 "orig": " ".join(text_tokens[i1:i2]),
#                 "repl": " ".join(cleaned_tokens[j1:j2]),
#                 "pos": (i1, i2)   # vị trí token trong gốc
#             })
#     return replacements, cleaned_text
def get_replacement_from_formatted_sentence(text, cleaned_text):
    cleaned = cleaned_text
    replacements = []
    
    matcher = difflib.SequenceMatcher(None, text, cleaned)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            replacements.append({
                "op": tag,              # replace / insert / delete
                "orig": text[i1:i2],    # đoạn gốc
                "repl": cleaned[j1:j2], # đoạn sửa
                "pos": (i1, i2)         # vị trí trong text gốc
            })
    return replacements, cleaned
MASK_TOKEN = "[MASK]"
MASK_TOKEN = "<mask>"
import torch

def fast_fill_mask(self, masked_text, top_k=1000, top_p=0.975):
    # raw model + tokenizer
    tokenizer = self.pipeline.tokenizer
    model = self.pipeline.model

    # tokenize
    inputs = tokenizer(masked_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    # find mask position
    mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

    # probs for the mask
    probs = torch.softmax(logits[0, mask_idx], dim=-1).squeeze(0)

    # only take top_k (faster than sorting all vocab)
    probs_top, idxs = torch.topk(probs, k=top_k)
    tokens = tokenizer.convert_ids_to_tokens(idxs)

    # build prediction dicts (like pipeline does)
    predictions = [
        {"token_str": tok, "score": float(score), "token": int(idx)}
        for tok, score, idx in zip(tokens, probs_top, idxs)
    ]

    # apply your nucleus (top_p) + "_" filtering
    filtered = []
    cum_p = 0.0
    for pred in predictions:
        if "_" in pred["token_str"]:
            continue
        cum_p += pred["score"]
        filtered.append(pred)
        if cum_p >= top_p:
            break

    return filtered

from name_checker import check_and_correct_word, spell_check_word
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AutoTokenizer, pipeline
from transformers import pipeline
from pyvi import ViTokenizer, ViPosTagger
import time
import torch

def filter_top_p(predictions, probs, top_p=0.975, topk=1000):
    # Convert to numpy for speed
    probs = np.array(probs)

    # take only topk first (assuming already sorted descending)
    probs = probs[:topk]
    preds = predictions[:topk]

    # mask out tokens with "_"
    mask = np.array(["_" not in pred['token_str'] for pred in preds])
    probs = probs[mask]
    preds = [p for p, keep in zip(preds, mask) if keep]

    # cumulative sum
    cum_probs = np.cumsum(probs)

    # find cutoff index
    cutoff = np.searchsorted(cum_probs, top_p)

    return preds[:cutoff+1]



def is_number(s):
    """Check if a string is a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False
    
class NER_Extractor:
    def __init__(self):
        self.ner_model_path = "NlpHUST/ner-vietnamese-electra-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.ner_model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.ner_model_path)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
    def __call__(self, query: str):
        """Call the NER extractor"""
        return self.extract_entities(self.get_raw_ner(query))

    def get_raw_ner(self, query: str):
        """Get raw NER results from the model"""
        return self.nlp(query)
    def extract_entities(self, ner_results):
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
        # get only person entities
        # entities = [ent for ent in entities if ent[1] == 'PERSON']
        return entities
class VNese_WordSegmenter:
    def __init__(self):
        self.tokenizer = ViTokenizer

    def __call__(self, text):
        return self.tokenizer.tokenize(text)
class VietnameseWordSegmenter:
    """
    A class for Vietnamese word segmentation using transformers.
    
    This class provides methods to segment Vietnamese text into words,
    handling subword tokens and word boundaries properly.
    """
    
    def __init__(self, model_name="NlpHUST/vi-word-segmentation"):
        """
        Initialize the Vietnamese Word Segmenter.
        
        Args:
            model_name (str): The name of the pre-trained model to use.
                            Default is "NlpHUST/vi-word-segmentation"
        """
        print(f"Loading Vietnamese Word Segmentation model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipeline = pipeline("token-classification", 
                                model=self.model, 
                                tokenizer=self.tokenizer)
        print("Model loaded successfully!")
    
    def segment(self, text):
        """
        Segment Vietnamese text into words.
        
        Args:
            text (str): The Vietnamese text to segment
            
        Returns:
            str: The segmented text with words separated by spaces and 
                 compound words connected by underscores
        """
        if not text or not text.strip():
            return ""
        
        # Get token classification results
        ner_results = self.pipeline(text.strip())
        
        # Process the results to create segmented text
        segmented_text = self._process_tokens(ner_results)
        
        return segmented_text.strip()
    
    def _process_tokens(self, ner_results):
        """
        Process the token classification results to create segmented text.
        
        Args:
            ner_results (list): List of token classification results
            
        Returns:
            str: Processed segmented text
        """
        segmented_text = ""
        
        for token_info in ner_results:
            word = token_info["word"]
            entity = token_info["entity"]
            
            if "##" in word:
                # Handle subword tokens (remove ## prefix)
                segmented_text += word.replace("##", "")
            elif entity == "I":
                # Inside entity - connect with underscore
                segmented_text += "_" + word
            else:
                # Beginning of entity or outside entity - add space
                segmented_text += " " + word
        
        return segmented_text
    
    def segment_batch(self, texts):
        return [self.segment(text) for text in texts]
    
    def get_word_boundaries(self, text):
        if not text or not text.strip():
            return []
        
        ner_results = self.pipeline(text.strip())
        word_info = []
        current_word = ""
        current_start = 0
        
        for i, token_info in enumerate(ner_results):
            word = token_info["word"]
            entity = token_info["entity"]
            start = token_info.get("start", 0)
            end = token_info.get("end", 0)
            
            if "##" in word:
                current_word += word.replace("##", "")
            elif entity == "I":
                current_word += "_" + word
            else:
                # Save previous word if exists
                if current_word:
                    word_info.append({
                        "word": current_word.strip(),
                        "start": current_start,
                        "end": end
                    })
                
                # Start new word
                current_word = word
                current_start = start
        
        # Add the last word
        if current_word:
            word_info.append({
                "word": current_word.strip(),
                "start": current_start,
                "end": ner_results[-1].get("end", len(text)) if ner_results else len(text)
            })
        
        return word_info
    
    def __call__(self, text):
        """
        Make the class callable - same as segment method.
        
        Args:
            text (str): The Vietnamese text to segment
            
        Returns:
            str: The segmented text
        """
        return self.segment(text) 
class BertSpellChecker:
    def __init__(self):
        self.model_name = "vinai/phobert-base-v2"
        self.pipeline = pipeline(
            task="fill-mask",
            model=self.model_name,
            tokenizer=self.model_name,
            torch_dtype=torch.float32,
            device=0
        )
        # phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.ner_extractor = NER_Extractor()
        self.segmenter_1 = VietnameseWordSegmenter()
        self.segmenter_2 = VNese_WordSegmenter()

    def __call__(self, text):
        return self.check_text(text)
    def mask_entities(self, text, entities):
        """Mask entities in the text with an alias"""
        entity_dict = {}
        for idx, entity in enumerate(entities):
            if entity[1] != 'PERSON':
                continue
            # alias = f"người_{idx}"
            alias = entity[0]
            entity_dict[alias] = entity[0]
            text = text.replace(entity[0], alias)
        print(f"Entity Dict: {entity_dict}")

        return text, entity_dict
    def loop_mask(self, text, mask_token="[MASK]", entities=None):
        """Loop through the text and mask entities"""
        words = text.split()
        masked_text = []
        data = []
        entities_list = list(entities.keys()) if entities else []
        for idx, word in enumerate(words):
            if word in removed_punctuation or word in entities_list:
                continue
            # skip words with more then 2 phenoms
            if "_" in word or len(word) < 2:
                continue
            masked_text = words[:idx] + [mask_token] + words[idx+1:]
            print(f"Masked Text: {' '.join(masked_text)}")
            masked_text = " ".join(masked_text)
            data.append({
                "text": masked_text,
                "index": idx,
                "word": word
            })
        return data
    def is_in_special_case(self, word):
        if is_number(word):
            return True
        if word.isdigit():
            return True 
        # remove punctuation
        word = remove_punctuation(word)
        if "," in word or "." in word:
            return True
        # measurement
        if word in ["kg", "cm", "mm", "m", "km", "g", "mg", "l", "ml", "tấn", "tạ", "yên", "đôla", "đồng"]:
            return True
        if is_datetime(word):
            return True
        if len(word) < 2:
            return True
        return False
    def check_single_word(self, sentence, entity_dict=None):
        mark_sentences = []
        print("sentence:", sentence)
        names = [ e.lower() for e in entity_dict] if entity_dict else []
        print("names:", names)
        for word in sentence.split(" "):
            word = word.lower()
            num_phenoms = word.count("_") + 1
            if word in names or word.replace("_", " ") in names:
                mark_sentences.append(word)
                continue
            if self.is_in_special_case(word):
                    # mark_sentences.append("<special_case>")
                    mark_sentences.append(word)
                    continue
            if num_phenoms > 2:
                mark_sentences.append(word)

                continue
            # split it into smaller parts
                sub_words = word.split("_")
                for s in sub_words:
                    if self.is_in_special_case(s):
                        mark_sentences.append(s)
                        continue
                    s, is_correct_word, suggestion = spell_check_word(s)
                    if not is_correct_word:
                        print(f"error in check_single_word (subword): {s} -> {suggestion}")
                        mark_sentences.append("<not_found_in_dict>")
                    else:
                        mark_sentences.append(s)
            else:
                word, is_correct_word, suggestion = spell_check_word(word)
                if not is_correct_word:
                    print(f"error in check_single_word: {word} -> {suggestion}")
                    mark_sentences.append("<not_found_in_dict>")
                else:
                    mark_sentences.append(word)
        mark_sentences = " ".join(mark_sentences)
        mark_sentences = clean_sentence(mark_sentences)
        
        print("final_mark_sentences_dict:", mark_sentences)
        # clean text 
        return mark_sentences
    
    def check_bert_mask(self, text, entity_dict=None, top_k=1000):
        list_of_segmented_text = text.split()
        list_of_masked = self.loop_mask(text, mask_token=MASK_TOKEN, entities=entity_dict)
        error_indices_list = []
        for item in list_of_masked:
                masked_text = item['text']
                # print(f"Masked Text: {masked_text}")
                index = item['index']
                word = item['word']
                # skip words with more than 2 phenoms 
                if "_" in word: 
                    continue
                entities_lower = [entity[0] for entity in entity_dict]
                if word.replace("_", " ") in entities_lower:
                    continue
                if self.is_in_special_case(word):
                    continue
                skip = False
                start_time = time.time()
                # print("input", masked_text)
                predictions = self.pipeline(masked_text.lower(), top_k=top_k)
                print(f"Pipeline took {time.time() - start_time:.2f} seconds")
                # print(f"Predictions for masked text '{masked_text}': {predictions}")
                probs = [pred['score'] for pred in predictions]
                # get only top_p = 0.95
                start_time = time.time()
                filtered_predictions = filter_top_p(predictions, probs, top_p=0.95, topk=top_k)
                end_time = time.time()
                # print(f"Filtering took {end_time - start_time:.2f} seconds")
                predictions = filtered_predictions
                if word.lower() not in [pred['token_str'].lower() for pred in predictions]:
                    if "_" in word:
                        continue
                    # check if it in the suggestion
                    for pred in predictions:
                        if word.lower() in pred['token_str'].lower().split("_"):
                            skip = True
                    if skip:
                        continue
                    max_suggestion = 5
                    sugg_text = ""
                    for s in predictions[:max_suggestion]:
                        sugg_text += f"{s['token_str']}*"
                    sugg_text = sugg_text.rstrip("*")
                    print("suggestion_0:", sugg_text)

                    error_indices_list.append(index)
        # replace error indices with error code <error_bert>
        for idx, text in enumerate(list_of_segmented_text):
            if idx in error_indices_list:
                list_of_segmented_text[idx] = "<not_found_in_bert>"
        marked_text = " ".join(list_of_segmented_text)
        marked_text = clean_sentence(marked_text)
        print("final_mark_sentences_bert:", marked_text)
        return marked_text
    def check_valid_entity_name(self, text, entities):
        for entity in entities:
            name = entity[0]
            word, is_correct, suggestion = check_and_correct_word(name)
            if not is_correct:
                print(f"error in check_valid_entity_name: {name} -> {suggestion}")
                # replace this name with <error_found_in_name>
                text = text.replace(name, f"<not_found_in_name>")
        text = clean_sentence(text)
        return text
    def seq2seq_correction(self, text):
        # split into sentences
        sentences = split_into_sentences(text)
        corrected_text = seq_2seq_correct_spelling(text)
        # corrected_text = clean_sentence(corrected_text)
        return corrected_text
    def check_text(self, ori_text):
        cleaned_text = clean_sentence(ori_text)
        if cleaned_text != ori_text:
            print("Text was cleaned.")
            replacement = get_replacement_from_formatted_sentence(ori_text, cleaned_text)
            
            return replacement[0], cleaned_text
        sentences = split_into_sentences(cleaned_text)

        err_list = []
        
        # upper_indices 
        # upper_indices = [i for i, c in enumerate(text) if c.isupper()]
        # first check
        
        # check for every sentence from here
        dict_results = []
        bert_results = []
        name_results = []
        seq2seq_results = []
        for text in sentences:
            entities = self.ner_extractor(text)
            text, entity_dict = self.mask_entities(text, entities)

            segmented_text = self.segmenter_1(text)
            dict_results.append(self.check_single_word(segmented_text, entity_dict))

            # bert_results.append(self.check_bert_mask(segmented_text, entity_dict, 200))
            seq2seq_results.append(self.seq2seq_correction(text))

            name_results.append(self.check_valid_entity_name(segmented_text, entities))
                    # list_of_segmented_words[index] = f"*{word}*"
                    # print(f"masked_text: {masked_text}, index: {index}, word: {word}, prediction: {predictions[0]['token_str']}")

        final_dict_paragraph = " ".join(dict_results)
        # final_bert_paragraph = " ".join(bert_results)
        final_seq2seq_paragraph = " ".join(seq2seq_results)
        final_name_paragraph = " ".join(name_results)
        # remove _ in paragrph
        final_dict_paragraph = replace_underscore_outside_tags(final_dict_paragraph)
        # final_bert_paragraph = replace_underscore_outside_tags(final_bert_paragraph)
        final_name_paragraph = replace_underscore_outside_tags(final_name_paragraph)
        print("after replacing underscores:")
        print("final_dict_paragraph:", final_dict_paragraph)
        # print("final_bert_paragraph:", final_bert_paragraph)
        print("final_name_paragraph:", final_name_paragraph)
        print("final_seq2seq_paragraph:", final_seq2seq_paragraph)
        # find diff
        dict_replacements, cleaned_dict = get_replacement_from_formatted_sentence(ori_text.lower(), final_dict_paragraph.replace('<not_found_in_dict>', '_').lower())
        # bert_replacements, cleaned_bert = get_replacement_from_formatted_sentence(ori_text.lower(), final_bert_paragraph.replace('<not_found_in_bert>', '_').lower())
        name_replacements, cleaned_name = get_replacement_from_formatted_sentence(ori_text.lower(), final_name_paragraph.replace('<not_found_in_name>', '_').lower())
        seq2seq_replacements, cleaned_seq2seq = get_replacement_from_formatted_sentence(ori_text, final_seq2seq_paragraph)
        print("cleaned_dict:", cleaned_dict)
        # print("cleaned_bert:", cleaned_bert)
        print("cleaned_name:", cleaned_name)
        print("cleaned_seq2seq:", cleaned_seq2seq)
        print("seq2seq_replacements:", seq2seq_replacements)
        print('dict_replacements:', dict_replacements)
        print('name_replacements:', name_replacements)
        # get final replacement and cleaned_text, remove duplicates
        merged_replacements = []
        for rep in dict_replacements:
            if is_comma_inside_number(rep['orig']):
                continue    
            merged_replacements.append(rep)
        
        filtered_bert = []
        if len(merged_replacements) > 0:
            for rep in seq2seq_replacements:
                for m in merged_replacements:
                    m_pos = m['pos']
                    if is_comma_inside_number(rep['orig']):
                        continue    
                    if is_inside(m_pos, rep['pos']):
                        continue 
                    else:
                        if not rep in merged_replacements and not rep in filtered_bert:
                            filtered_bert.append(rep)
        else:
            filtered_bert = seq2seq_replacements
        merged_replacements.extend(filtered_bert)
        filtered_name = []
       
        for rep in name_replacements:
            if merged_replacements:
                for m in merged_replacements:
                    m_pos = m['pos']
                    if is_comma_inside_number(rep['orig']):
                        continue    
                    if is_inside(m_pos, rep['pos']):
                        continue
                    else:
                        if not rep in merged_replacements and not rep in filtered_name:
                            filtered_name.append(rep)
            else:
                filtered_name = name_replacements
                break
        merged_replacements.extend(filtered_name)
        print("merged_replacements:", merged_replacements   )
        final_cleaned_text = ori_text
        # character_list = ori_text.split(" ")
        character_list = list()
        print("character_list:", character_list)
        for c in ori_text:
            character_list.append(c)
        character_list_copy = copy.copy(character_list)

        for r in merged_replacements:
            op = r['op']
            orig = r['orig']
            repl = r['repl']
            pos = r['pos']
            start, end = pos
            repl_tokens = list(repl)
            
                # số lượng token gốc
            orig_len = end - start
            # print("orig_len:", orig_len)
            # mark bằng "_" giữ nguyên độ dài
        
            character_list_copy[start:end] = ["_"] * orig_len

        final_cleaned_text = "".join(character_list_copy)
        print("final_cleaned_text:", final_cleaned_text)
        return merged_replacements, final_cleaned_text

if __name__ == "__main__":
    spell_checker = BertSpellChecker()
    TXT = "Thủ tướng Phạm Minh Chính đax tham gia cùng tổng bí thư Tô Lâm"
    text = TXT
    marked_text, errors = spell_checker(text)
    print(f"Corrected Text: {marked_text}")
    print(f"Errors: {errors}")
