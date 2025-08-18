import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from mis_detection import sentence_prediction
import re 
class SpellCheckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vietnamese Spell Checker Demo")
        self.root.geometry("1000x600")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Vietnamese Spell Checker", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left side - Original Text
        left_frame = ttk.LabelFrame(main_frame, text="Original Text", padding="10")
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        
        # Original text area
        self.original_text = scrolledtext.ScrolledText(
            left_frame,
            wrap=tk.WORD,
            width=40,
            height=20,
            font=("Arial", 21)
        )
        self.original_text.grid(row=0, column=0, sticky="nsew")
        self.original_text.insert('1.0', "Paste your Vietnamese text here...\n\nExample:\nTổng bí thư Tô Lâm và thủ tướng Phạm Minh Chính có bài phát biểu tại hội nghị.")

        # Buttons frame
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        
        # Clear button
        clear_btn = ttk.Button(buttons_frame, text="Clear", command=self.clear_text)
        clear_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        # Check spelling button
        check_btn = ttk.Button(buttons_frame, text="Check Spelling", 
                              command=self.check_spelling, style="Accent.TButton")
        check_btn.grid(row=0, column=1, sticky="ew")
        
        # Right side - Warnings
        right_frame = ttk.LabelFrame(main_frame, text="Potential Misspellings", padding="10")
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # Warnings text area
        self.warnings_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            width=40,
            height=20,
            font=("Arial", 21),
            state=tk.DISABLED,
            bg="#f8f8f8"
        )
        self.warnings_text.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # Configure tags for text highlighting
        self.warnings_text.tag_configure("error", foreground="red", font=("Arial", 21, "bold"))
        self.warnings_text.tag_configure("suggestion", foreground="green", font=("Arial", 21, "italic"))
        self.warnings_text.tag_configure("normal", foreground="black", font=("Arial", 21))
        self.warnings_text.tag_configure("underline", underline=True, foreground="red", font=("Arial", 21, "bold"))
        self.warnings_text.tag_configure("correct", foreground="black", font=("Arial", 21))
    
    def clear_text(self):
        """Clear the original text area"""
        self.original_text.delete('1.0', tk.END)
        self.warnings_text.config(state=tk.NORMAL)
        self.warnings_text.delete('1.0', tk.END)
        self.warnings_text.config(state=tk.DISABLED)
        self.status_var.set("Text cleared")
    
    def check_spelling(self):
        """Check spelling in the original text"""
        text = self.original_text.get('1.0', tk.END).strip()
        
        if not text:
            self.status_var.set("Please enter some text to check")
            return
        
        # Update status
        self.status_var.set("Checking spelling...")
        
        # Run spell check in a separate thread to avoid UI freezing
        thread = threading.Thread(target=self._perform_spell_check, args=(text,))
        thread.daemon = True
        thread.start()
    
    def _perform_spell_check(self, text):
        """Perform spell checking"""
        # Clear warnings area
        self.warnings_text.config(state=tk.NORMAL)
        self.warnings_text.delete('1.0', tk.END)
        
        # Add header
        self.warnings_text.insert(tk.END, "🔍 Spell Check Results\n", "normal")
        self.warnings_text.insert(tk.END, "=" * 40 + "\n", "normal")
        
        # Get spell check results
        potential_errors = self._detect_misspellings_placeholder(text)
        
        if potential_errors:
            self.warnings_text.insert(tk.END, f"Found {len(potential_errors)} potential misspelling(s):\n", "normal")
            
            # Display original text with underlined misspellings for each sentence
            lines = text.split('. ')
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # self.warnings_text.insert(tk.END, f"Sentence {line_num + 1}:\n", "normal")
     

                # Get errors for this line
                line_errors = [error for error in potential_errors if error[1] == line_num]
                
                if line_errors:
                    # Display the sentence with underlined errors
                    self._display_sentence_with_errors(line, line_errors)
                    
                    # # Display suggestions
                    # self.warnings_text.insert(tk.END, "\nSuggestions:\n", "normal")
                    # for i, (word, _, suggestions) in enumerate(line_errors, 1):
                    #     self.warnings_text.insert(tk.END, f"  {i}. ", "normal")
                    #     self.warnings_text.insert(tk.END, f"'{word}'", "error")
                    #     self.warnings_text.insert(tk.END, " → ", "normal")
                    #     if isinstance(suggestions, list) and suggestions:
                    #         self.warnings_text.insert(tk.END, f"{suggestions[0]}", "suggestion")
                    #     else:
                    #         self.warnings_text.insert(tk.END, f"{suggestions}", "suggestion")
                    #     self.warnings_text.insert(tk.END, "\n", "normal")
                else:
                    # Display sentence without errors
                    self.warnings_text.insert(tk.END, line, "correct")
                
                self.warnings_text.insert(tk.END, "\n", "normal")
        else:
            self.warnings_text.insert(tk.END, "✅ No misspellings detected!\n", "normal")
        
        self.warnings_text.config(state=tk.DISABLED)
        
        # Update status on main thread
        self.root.after(0, lambda: self.status_var.set("Spell check completed"))
    
    def _display_sentence_with_errors(self, sentence, errors):
        """Display sentence with underlined misspelled words"""
        from pyvi import ViTokenizer
        
        # Tokenize the sentence to match the error indices
        # tokenized = ViTokenizer.tokenize(sentence)
        # words = tokenized.split()
        # Create a set of error word indices for quick lookup
        print(f"Errors: {errors}")
        error_words = {error[1].replace("_", " ").lower() for error in errors}
        for er in error_words:
            print("Replacing error word:", er, "to", er.replace(" ", "_").lower())
            sentence = sentence.replace(er, er.replace(" ", "_").lower()) 
        words = sentence.split()
        for i, word in enumerate(words):
            print(f"Word: {word}, Index: {i}")
            if "_" in word or word in error_words:
                # This word has an error - underline it
                word = word.replace("_", " ").lower()
                self.warnings_text.insert(tk.END, word, "error")
            else:
                # This word is correct
                word = word.replace("_", " ").lower()
                self.warnings_text.insert(tk.END, word, "correct")
            
            # Add space between words (except for the last word)
            if i < len(words) - 1:
                self.warnings_text.insert(tk.END, " ", "correct")
    
    def _detect_misspellings_placeholder(self, text):
        """
        PLACEHOLDER FUNCTION for spell checking detection
        
        This function simulates misspelling detection and will be replaced
        with the actual implementation in the next phase.
        
        Args:
            text (str): The text to check for misspellings
            
        Returns:
            list: List of tuples (misspelled_word, line_number, suggestions)
        """
        
        # Simulate some common misspellings for demonstration
        potential_errors = []
        lines = text.split('. ')
        
      
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            err = sentence_prediction(line, top_k=1000)
            if err:
                for idx, word, suggestions in err:
                    potential_errors.append((word, line_num, suggestions))
                    
        
        # For demo purposes, limit to first few errors
        return potential_errors[:5]

def main():
    root = tk.Tk()
    app = SpellCheckApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
