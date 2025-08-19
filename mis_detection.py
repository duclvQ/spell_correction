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
from name_checker import check_and_correct_word
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AutoTokenizer, pipeline
from transformers import pipeline
from pyvi import ViTokenizer, ViPosTagger
import time
import torch


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
    
class BertSpellChecker:
    def __init__(self):
        self.model_name = "vinai/phobert-base-v2"
        self.pipeline = pipeline(
            task="fill-mask",
            model=self.model_name,
            tokenizer=self.model_name,
            torch_dtype=torch.float16,
            device=0
        )
        # phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.ner_extractor = NER_Extractor()
        self.segmenter = VNese_WordSegmenter()

    def __call__(self, text):
        return self.sentence_prediction(text)
    def mask_entities(self, text, entities):
        """Mask entities in the text with an alias"""
        entity_dict = {}
        for idx, entity in enumerate(entities):
            if entity[1] != 'PERSON':
                continue
            alias = f"người_{idx}"
            entity_dict[alias] = entity[0]
            text = text.replace(entity[0], alias)
        print(f"Entity Dict: {entity_dict}")
        return text, entity_dict
    def loop_mask(self, text, mask_token="<mask>", entities=None):
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
            
    def sentence_prediction(self, text, top_k=200):
        entities = self.ner_extractor(text)

        text, entity_dict = self.mask_entities(text, entities)
    
        segmented_text = self.segmenter(text)
        print(f"Segmented Text: {segmented_text}")

        list_of_segmented_words = segmented_text.split()
        list_of_masked = self.loop_mask(segmented_text, mask_token="<mask>", entities=entity_dict)
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
            predictions = self.pipeline(masked_text, top_k=top_k)
            if word.lower() not in [pred['token_str'].lower() for pred in predictions]:
                err_list.append((index, word, predictions[0]['token_str']))
                list_of_segmented_words[index] = f"*{word}*"
                print(f"masked_text: {masked_text}, index: {index}, word: {word}, prediction: {predictions[0]['token_str']}")
        checked_text = " ".join(list_of_segmented_words)
        for index, word, correction in err_list:
            print(f"Error found at index {index}: {word} -> {correction}")
        for alias, name in entity_dict.items():
            word, is_correct, correction = check_and_correct_word(name)
            print(f"Checking word: {name}, Corrected: {word}, Is Correct: {is_correct}, Suggestion: {correction}")
            if not is_correct:
                
                checked_text = checked_text.replace(alias, f"*{name}*")
        # inverser the entity masking
        for alias, original in entity_dict.items():
            checked_text = checked_text.replace(alias, original)
        # remove underscores from the checked text
        checked_text = checked_text.replace("_", " ")
        return checked_text, err_list


if __name__ == "__main__":
    spell_checker = BertSpellChecker()
    text = """Thủ tướng Phạ Minh Chính cho rằng nhiều công trình khởi công ngày 19/8 sẽ trở thành biểu tượng mới, được thế giới ngưỡng mộ, mở thêm không gian văn hóa để nhân dân thụ hưởng.
Phát biểu tại lễ khởi công và khánh thành 250 công trình, dự án trên cả nước sáng 19/8, Thủ tướng nhấn mạnh các dự án lần này có ý nghĩa chiến lược trong phát triển hạ tầng, tạo động lực thúc đẩy kinh tế - xã hội, đồng thời thu hút mạnh mẽ nguồn lực tư nhân.
Trong tổng số 250 công trình, có 89 dự án được khánh thành với tổng vốn đầu tư khoảng 220.000 tỷ đồng, bao gồm 208 km đường cao tốc, nâng tổng chiều dài đường bộ cao tốc cả nước lên gần 2.500 km. Nhiều dự án quy mô lớn được đưa vào sử dụng như cầu Rạch Miễu 2, Nhà máy thủy điện Trị An mở rộng, Hòa Bình mở rộng, Bệnh viện Ung bướu Nghệ An 1.000 giường, trụ sở Bộ Công an và Trung tâm tài chính quốc tế Saigon Marina.
"""

    corrected_text, errors = spell_checker(text)
    print(f"Corrected Text: {corrected_text}")
    print(f"Errors: {errors}")
