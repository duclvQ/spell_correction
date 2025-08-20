from transformers.models.canine.tokenization_canine import MASK
import numpy as np
import time 
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
removed_punctuation = [
    "!", "?", ".", ",", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*",
    "+", "-", "=", "~", "`"
]

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

from name_checker import check_and_correct_word
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
        """
        Segment multiple texts at once.
        
        Args:
            texts (list): List of Vietnamese texts to segment
            
        Returns:
            list: List of segmented texts
        """
        return [self.segment(text) for text in texts]
    
    def get_word_boundaries(self, text):
        """
        Get detailed word boundary information.
        
        Args:
            text (str): The Vietnamese text to analyze
            
        Returns:
            list: List of dictionaries containing word information
        """
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
        self.segmenter = VietnameseWordSegmenter()

    def __call__(self, text):
        return self.sentence_prediction(text)
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
            masked_text = " ".join(masked_text)
            data.append({
                "text": masked_text,
                "index": idx,
                "word": word
            })
        return data
            
    def sentence_prediction(self, text, top_k=500):
        entities = self.ner_extractor(text)

        text, entity_dict = self.mask_entities(text, entities)
    
        segmented_text = self.segmenter(text)
        print(f"Segmented Text: {segmented_text}")

        list_of_segmented_words = segmented_text.split()
        list_of_masked = self.loop_mask(segmented_text, mask_token=MASK_TOKEN, entities=entity_dict)
        err_list = []
        for item in list_of_masked:
            masked_text = item['text']
            index = item['index']
            word = item['word']
            # skip if the word is already in the entity list
            if word in [entity[0] for entity in entities]:
                continue
            # skip if the word is a number
            if is_number(word):
                continue
        
            skip = False
            start_time = time.time()
            predictions = self.pipeline(masked_text, top_k=top_k)
            
            print(f"Pipeline took {time.time() - start_time:.2f} seconds")
            # print(f"Predictions for masked text '{masked_text}': {predictions}")
            probs = [pred['score'] for pred in predictions]
            # get only top_p = 0.95
            start_time = time.time()
            filtered_predictions = filter_top_p(predictions, probs, top_p=0.975, topk=top_k)
            end_time = time.time()
            print(f"Filtering took {end_time - start_time:.2f} seconds")
            predictions = filtered_predictions
            if word.lower() not in [pred['token_str'].lower() for pred in predictions]:
                # check if it in the suggestion
                for pred in predictions:
                    if word.lower() in pred['token_str'].lower().split("_"):
                        skip = True
                if skip:
                    continue
                suggestion = ["".join(predictions[i]['token_str']) for i in range(1)]
                err_list.append((index, word, suggestion))
                list_of_segmented_words[index] = f"*{word}*"
                print(f"masked_text: {masked_text}, index: {index}, word: {word}, prediction: {predictions[0]['token_str']}")
        
        checked_text = " ".join(list_of_segmented_words)
        for index, word, correction in err_list:
            print(f"Error found at index {index}: {word} -> {correction}")
        for alias, name in entity_dict.items():
            word, is_correct, correction = check_and_correct_word(name)
            # upper first phenon
            word_list = [word.capitalize() for word in name.split("_")]
            word = " ".join(word_list)
            print(f"Checking word: {name}, Corrected: {word}, Is Correct: {is_correct}, Suggestion: {correction}")
            if not is_correct:
                under_name = name.replace(" ", "_")
                checked_text = checked_text.replace(under_name, f"*{name}*")
                err_list.append((0, name, correction))
        # inverser the entity masking
        for alias, original in entity_dict.items():
            checked_text = checked_text.replace(alias, original)
        # remove underscores from the checked text
        checked_text = checked_text.replace("_", " ")
        return checked_text, err_list


if __name__ == "__main__":
    spell_checker = BertSpellChecker()
    text = """Thủ tướng Phạn Minh Chính cho rằng nhiều công trình khởi công ngày 19/8 sẽ trở thành biểu tượng mới, được thế giới ngưỡng mộ, mở thêm không gian văn hóa để nhân dân thụ hưởng.
Phát biểu tại lễ khởi công và khánh thành 250 công trình, dự án trên cả nước sáng 19/8, Thủ tướng nhấn mạnh các dự án lần này có ý nghĩa chiến lược trong phát triển hạ tầng, tạo động lực thúc đẩy kinh tế - xã hội, đồng thời thu hút mạnh mẽ nguồn lực tư nhân.
Trong tổng số 250 công trình, có 89 dự án được khánh thành với tổng vốn đầu tư khoảng 220.000 tỷ đồng, bao gồm 208 km đường cao tốc, nâng tổng chiều dài đường bộ cao tốc cả nước lên gần 2.500 km. Nhiều dự án quy mô lớn được đưa vào sử dụng như cầu Rạch Miễu 2, Nhà máy thủy điện Trị An mở rộng, Hòa Bình mở rộng, Bệnh viện Ung bướu Nghệ An 1.000 giường, trụ sở Bộ Công an và Trung tâm tài chính quốc tế Saigon Marina.
"""

    corrected_text, errors = spell_checker(text)
    print(f"Corrected Text: {corrected_text}")
    print(f"Errors: {errors}")
