from sklearn.feature_extraction.text import CountVectorizer

weights = {}

class mem_arch:
    def __init__(self, input_text):
        self.input_text = input_text
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit([input_text])
        self.vocab = self.vectorizer.get_feature_names_out()
        self.word_counts = self.vectorizer.transform([input_text]).toarray().flatten()
        self.mem = []

    def process_word_vectors(self):
        for word, count in zip(self.vocab, self.word_counts):
            vec = [0] * len(self.vocab)
            vec[self.vectorizer.vocabulary_[word]] = count
            self.mem.append([word] + vec)

    def cron_weight(self): 
        n = len(self.mem)
        for i, item in enumerate(self.mem):
            if i == 0:
                weights[tuple(item)] = n
            elif i == len(self.mem) - 1:
                weights[tuple(item)] = 1
        return weights

input_text = "what is a human computer interface"
word_vector_processor = mem_arch(input_text)
word_vector_processor.process_word_vectors()
print(word_vector_processor.mem)
print(word_vector_processor.cron_weight())
