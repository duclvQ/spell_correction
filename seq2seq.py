# load model 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("/Users/leduc/Downloads/checkpoint-50000")
device = "mps"
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
def seq_2seq_correct_spelling(input_text):
    model.eval() 
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=64, num_beams=10, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
if __name__ == "__main__":
    # test
    input_text = "Ngày Chủ tịch Hồ Chí Minh đọc Tuyên ngôn độc lập, cụ Năm mới là thiếu nữ 18 tuổi, ho vào dòng người tới sân bóng xã dự lễ mít tinh. Bàn con truyền tai nhau câu chuyện về một người con Nghệ An nay trở thành lãnh tụ mới của đất nước, không giấu niềm tự hào."
    model.eval() 
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    outputs = model.generate(input_ids, max_length=64, num_beams=10, early_stopping=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))