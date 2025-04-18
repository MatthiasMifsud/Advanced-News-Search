import customtkinter as ctk
from fetcher import News_API
from processing import Text_Processor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NewsSearchApp:
    def __init__(self, api_key, source_list):
        self.api_key = api_key
        self.source_list = source_list
        self.news_api = News_API(api_key, source_list)
        self.text_processor = Text_Processor()
        self.vectorizer = TfidfVectorizer()
        self.threshold_value = 0.15

        self.init_ui()

    def init_ui(self):
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.geometry("1000x1000")
        self.root.title("News Search Engine")

        self.create_frames()
        self.create_textbox_label()
        self.create_textbox()
        self.create_label()
        self.create_slider()
        self.create_button()
        self.create_outputbox_label()
        self.create_outputbox()

        self.root.mainloop()

    def create_frames(self):
        self.textFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.textFrame.pack(side="top", pady=(10, 10))

        self.messageFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.messageFrame.pack(side="top", pady=(10, 10))

        self.buttonFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.buttonFrame.pack(side="bottom", pady=(10, 50))

        self.sliderFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.sliderFrame.pack(side="top", pady=(10, 10))

        self.outputFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.outputFrame.pack(side="bottom", pady=(10, 10))

    def create_label(self):
        self.message = ctk.CTkLabel(self.messageFrame, text="", width=400, height=30)
        self.message.pack(pady=10, padx=10)

    def create_textbox_label(self):
        self.textboxLabel = ctk.CTkLabel(self.textFrame, text="Describe what news you're looking for below:", font=ctk.CTkFont(size=18, weight="bold"), text_color=("#2E2E2E", "#F5F5F5"))
        self.textboxLabel.pack(pady=(10, 0))

    def create_textbox(self):
        self.textbox = ctk.CTkTextbox(self.textFrame, width=400, height=200)
        self.textbox.pack(pady=10, padx=10)

    def create_button(self):
        search_button = ctk.CTkButton(self.buttonFrame, text="Search", command=self.runner)
        search_button.grid(row=0, column=0, padx=5, pady=5)

        clear_button = ctk.CTkButton(self.buttonFrame, text="Clear", command=self.clear_all)
        clear_button.grid(row=0, column=1, padx=5, pady=5)

        toggle_button = ctk.CTkButton(self.buttonFrame, text="Toggle Theme", command=self.toggle_appearance)
        toggle_button.grid(row=0, column=2, padx=5, pady=5)

    def slider_callback(self, value):
        self.threshold_value = float(value)
        self.threshold_label.configure(text=f"Adjust Fallback Threshold | {int(self.threshold_value * 100)}%", font=ctk.CTkFont(size=12, weight="bold"), text_color=("#2E2E2E", "#F5F5F5"))

    def create_slider(self):
        self.threshold_slider = ctk.CTkSlider(self.sliderFrame, from_=0, to=1, command=self.slider_callback)
        self.threshold_slider.pack()

        self.threshold_label = ctk.CTkLabel(self.sliderFrame, text=f"Adjust Fallback Threshold | {int(self.threshold_value * 100)}%", font=ctk.CTkFont(size=12, weight="bold"), text_color=("#2E2E2E", "#F5F5F5"))
        self.threshold_label.pack()

    def create_outputbox_label(self):
        self.outputboxLabel = ctk.CTkLabel(self.outputFrame, text="Results:", font=ctk.CTkFont(size=18, weight="bold"), text_color=("#2E2E2E", "#F5F5F5"))
        self.outputboxLabel.pack(pady=(10, 0))

    def create_outputbox(self):
        self.outputbox = ctk.CTkTextbox(self.outputFrame, width=800, height=300, wrap="word")
        self.outputbox.pack(pady=10, padx=10)

        self.outputbox.tag_config("green", foreground="green")
        self.outputbox.tag_config("orange", foreground="orange")
        self.outputbox.tag_config("gray", foreground="gray")

        self.outputbox.bind("<Key>", lambda e: "break")  
        self.outputbox.bind("<Button-1>", lambda e: self.outputbox.focus_set())  
        self.outputbox.bind("<Control-c>", lambda e: self.root.clipboard_append(self.outputbox.selection_get()))
        self.outputbox.bind("<Command-c>", lambda e: self.root.clipboard_append(self.outputbox.selection_get()))

    def clear_all(self):
        self.textbox.delete("1.0", "end")
        self.outputbox.configure(state="normal")
        self.outputbox.delete("1.0", "end")
        self.outputbox.configure(state="disabled")
        self.message.configure(text="")

    def similarity(self, texts, user_text):
        corpus = list(texts.values())
        corpus.append(user_text)

        tfid_matrix = self.vectorizer.fit_transform(corpus)
        cos_sim_matrix = cosine_similarity(tfid_matrix[-1], tfid_matrix[:-1])

        threshold = max(np.percentile(cos_sim_matrix, 90), 0.1)

        self.outputbox.configure(state="normal")
        self.outputbox.delete("1.0", "end")
        self.outputbox.insert("end", f"Fallback Threshold: {int(self.threshold_value * 100)}%   |   Threshold: {int(threshold * 100)}%\n\n")

        matched_links = []

        for num, similarity in enumerate(cos_sim_matrix[0]):
            if similarity > threshold and similarity > self.threshold_value:
                link = list(texts.keys())[num]
                matched_links.append(link)

                # Color tag
                if similarity > 0.30:
                    tag = "green"
                elif similarity > 0.15:
                    tag = "orange"
                else:
                    tag = "gray"

                self.outputbox.insert("end", "--------------------------------------------------------\n")
                self.outputbox.insert("end", f"Link: {link}\n", tag)
                self.outputbox.insert("end", f"Similarity: {int(similarity * 100)}%\n", tag)
                self.outputbox.insert("end", "--------------------------------------------------------\n\n")

        self.outputbox.configure(state="disabled")
        return matched_links
    
    def runner(self):
        self.news_api.parallel_news_fetching()  

        try:
            user_input = self.textbox.get("1.0", "end-1c").strip()
            if not user_input:
                self.message.configure(text="Please enter something!", text_color="red")
                return

            processed_user_text = self.text_processor.preprocessing(user_input)

        except Exception as e:
            self.message.configure(text=f"Failed to load/process your description! Error: {e}", text_color="red")
            return

        processed_texts = {}
        for article in self.news_api.articles:
            title = article.get('title', '')
            description = article.get('description', '')
            combined_text = f"{title} {description}"

            processed_texts[article['url']] = self.text_processor.preprocessing(combined_text)

        if not processed_texts:
            self.message.configure(text="No articles matched your description!", text_color="red")
            return

        matched_links = self.similarity(processed_texts, processed_user_text)
        self.message.configure(
            text=f"Found {len(matched_links)} out of {len(processed_texts)} articles which match your description!",
            text_color="green"
        )
    
    def toggle_appearance(self):
        current = ctk.get_appearance_mode()
        new_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)

