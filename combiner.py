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

        self.threshold_value = 0.15
        self.vectorizer = TfidfVectorizer()

        self.init_ui()

    def init_ui(self):
        # GUI Setup
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.geometry("1000x1000")
        self.root.title("News Search")

        self.create_frames()
        self.create_textbox()
        self.create_slider()
        self.create_button()
        self.create_outputbox()

        self.root.mainloop()

    def create_frames(self):
        self.textFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.textFrame.pack(side="top", pady=(10, 10))

        self.buttonFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.buttonFrame.pack(side="bottom", pady=(10, 50))

        self.sliderFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.sliderFrame.pack(side="top", pady=(10, 10))

        self.outputFrame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.outputFrame.pack(side="bottom", pady=(10, 10))

    def search(self):
        return self.textbox.get("1.0", "end-1c").strip()

    def create_textbox(self):
        self.textbox = ctk.CTkTextbox(self.textFrame, width=400, height=200)
        self.textbox.pack(pady=10, padx=10)

    def create_button(self):
        search_button = ctk.CTkButton(self.buttonFrame, text="Search", command=self.runner)
        search_button.grid(row=0, column=0, padx=5, pady=5)

    def slider_callback(self, value):
        self.threshold_value = float(value)
        self.threshold_label.configure(text=f"Threshold = {self.threshold_value:.2f}")

    def create_slider(self):
        self.threshold_slider = ctk.CTkSlider(self.sliderFrame, from_=0, to=1, command=self.slider_callback)
        self.threshold_slider.pack()

        self.threshold_label = ctk.CTkLabel(self.sliderFrame, text=f"Threshold = {self.threshold_value:.2f}")
        self.threshold_label.pack()

    def make_readonly(self):
        self.outputbox.bind("<Key>", lambda event: "break")  # Disable keyboard input
        self.outputbox.bind("<Button-1>", lambda event: "break")  # Disable mouse interaction


    def create_outputbox(self):
        self.outputbox = ctk.CTkTextbox(self.outputFrame, width=800, height=300, wrap="word")
        self.outputbox.pack(pady=10, padx=10)
        self.make_readonly()

    def tfid_calc(self, texts, user_text):
        corpus = list(texts.values())
        corpus.append(user_text)

        tfid_matrix = self.vectorizer.fit_transform(corpus)
        cos_sim_matrix = cosine_similarity(tfid_matrix[-1], tfid_matrix[:-1])

        threshold = np.percentile(cos_sim_matrix, 90)  # Calculate adaptive threshold

        self.outputbox.delete("1.0", "end")  # Clear previous output
        self.outputbox.insert("end", f"Threshold: {threshold:.4f}\n\n")

        for num, similarity in enumerate(cos_sim_matrix[0]):
            # Compare with both adaptive threshold and user-adjusted slider threshold
            if similarity > threshold and similarity > self.threshold_value:
                link = list(texts.keys())[num]
                self.outputbox.insert("end", f"Link: {link}, Similarity: {similarity:.4f}\n\n")

    def runner(self):
        self.news_api.parallel_news_fetching()  # Fetch articles in parallel

        try:
            user_input = self.search()
            if not user_input:
                print("Please enter a text")
                return

            processed_user_text = self.text_processor.preprocessing(user_input)

        except Exception as e:
            print(f"Failed to load/process the text. Error: {e}")
            return

        processed_texts = {}
        for article in self.news_api.articles:
            title = article.get('title', '')
            description = article.get('description', '')
            combined_text = f"{title} {description}"

            processed_texts[article['url']] = self.text_processor.preprocessing(combined_text)

        if not processed_texts:
            print("No articles were fetched to compare with user text.")
            return

        print(f"Extracted {len(processed_texts)} articles.")
        self.tfid_calc(processed_texts, processed_user_text)


# Example initialization:
# app = NewsSearchApp(api_key="your_api_key", source_list=["bbc-news", "cnn"])
