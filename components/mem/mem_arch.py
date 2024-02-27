from word_vectors.word_vecs import model, vocab

class mem_arch:
    def __init__(self, input_text):
        self.input_text = input_text
        self.spl = input_text.split()
        self.filt = [word for word in self.spl if word in vocab]
        self.mem = []

    def process_word_vectors(self):
        for word in self.filt:
            vec = model.wv[word]
            self.mem.append([word] + vec.tolist())

    def cron_weight(self): 
        weights = {} 
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
