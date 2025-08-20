"""
Vietnamese Word Segmenter Class

A ready-to-use class for Vietnamese word segmentation using transformers.
This class provides methods to segment Vietnamese text into words,
handling subword tokens and word boundaries properly.

Author: Generated for spell correction project
Date: August 20, 2025
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


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


def main():
    """
    Example usage of the VietnameseWordSegmenter class.
    """
    # Initialize the segmenter
    segmenter = VietnameseWordSegmenter()
    
    # Example texts to segment
    examples = [
        "Trước đây, giá đất ở khu vực này chỉ 2,7 triệu đồng mỗi m2.",
        "Trung tâm Dự báo Khí tượng Thủy văn quốc gia cho biết lúc 7h hôm nay",
        "Trường đại học bách khoa hà nội là một trong những trường hàng đầu"
    ]
    
    print("=== Vietnamese Word Segmentation Demo ===\n")
    
    for i, text in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Original: {text}")
        
        # Basic segmentation
        segmented = segmenter.segment(text)
        print(f"Segmented: {segmented}")
        
        # Get word boundaries (optional)
        boundaries = segmenter.get_word_boundaries(text)
        print(f"Word boundaries: {len(boundaries)} words found")
        
        print("-" * 50)
    
    # Batch processing example
    print("\n=== Batch Processing ===")
    batch_results = segmenter.segment_batch(examples)
    for i, result in enumerate(batch_results, 1):
        print(f"Batch {i}: {result}")
    
    # Using the class as a callable
    print("\n=== Using as Callable ===")
    callable_result = segmenter("Tôi yêu Việt Nam")
    print(f"Callable result: {callable_result}")


if __name__ == "__main__":
    main()
