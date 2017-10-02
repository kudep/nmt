from gensim.models.wrappers import FastText
import collections
#todo: add merge

__all__ = ["EmbeddingsGenerator"]

class EmbeddingsGenerator():
    def __init__(self, gen_model = "model.bin"):
        self.model = FastText.load_fasttext_format(gen_model)
        self.embeddings = None
        self.emb_vocab = None
        self.miss_vocab = None
        self.loaded_vocab = None
    def _gen_w2v(self):
        """Process generates embeddings by a loaded vocab."""
        assert not self.loaded_vocab is None
        self.embeddings = dict()
        self.emb_vocab = list()
        self.miss_vocab = list()
        for word in self.loaded_vocab:
            try:
                self.embeddings[word] = self.model[word]
                self.emb_vocab.append(word)
            except KeyError:
                self.miss_vocab.append(word)
                continue

    def load_vocab_from_file(self, vocabfile, tag_list = None):
        """Process loads vocab from file."""
        self._data_reset()
        self.loaded_vocab = list()
        with open(vocabfile) as vocf:
            for line in vocf.readlines():
                line = line.split()
                self.loaded_vocab.append(line[0])
                assert len(line) == 1
        self._insert_tag_into_vocab(tag_list)

    def load_vocab_from_corpus_file(self, corpus_file, max_words = None, tag_list = None):
        """Process loads vocab from corpus file."""
        with open(corpus_file) as corpf:
            words = corpf.read().split()
        self.load_vocab_from_list_of_words(words, max_words,tag_list)


    def load_vocab_from_list_of_rows(self, rowslist, max_words = None, tag_list = None):
        """Process loads vocab from list of rows."""
        entire_text = ""
        for row in rowslist: entire_text+=" " + row
        words = entire_text.split()
        self.load_vocab_from_list_of_words(words, max_words,tag_list)



    def load_vocab_from_list_of_words(self, wordlist, max_words = None, tag_list = None):
        """Process loads vocab from list of text."""
        self._data_reset()
        self.loaded_vocab = list()
        count = collections.Counter(wordlist).most_common(max_words)

        for word, _ in count:
            self.loaded_vocab.append(word)
        self._insert_tag_into_vocab(tag_list)


    def _data_reset(self):
        """Process resets all datas."""
        self.embeddings = None
        self.emb_vocab = None
        self.miss_vocab = None

    def _insert_tag_into_vocab(self,tag_list):
        """Process inserts addition tags."""
        #tag insert first in list, for example ["<unk>", "<s>", "</s>"]
        if tag_list:
            vocab_without_tag=[]
            for word in self.loaded_vocab:
                if word in tag_list:
                    continue
                else:
                    vocab_without_tag.append(word)
            self.loaded_vocab= tag_list + vocab_without_tag

    def _check_gen_w2v(self):
        """Process verificates generation data."""
        return (self.embeddings is None) or (self.miss_vocab is None) or (self.emb_vocab is None)

    def get_all_data(self):
        """Process returns generation data."""
        if (self._check_gen_w2v()):
            self._gen_w2v()
        return self.embeddings, self.miss_vocab, self.emb_vocab

    def save_embeddings(self, embedfile):
        """Process saves embeddings into file."""
        if (self._check_gen_w2v()):
            self._gen_w2v()
        with open(embedfile, 'wt') as embf:
            for word in self.emb_vocab:
                embf.write(word + ' ')
                vect = ' '.join(map(str, self.embeddings[word]))
                embf.write(vect)
                embf.write('\n')

    def save_embedded_vocab(self, vocabfile):
        """Process saves embedded vocab into file."""
        if (self._check_gen_w2v()):
            self._gen_w2v()
        with open(vocabfile, 'wt') as vocf:
            for word in self.emb_vocab:
                vocf.write(word + '\n')

    def merge_embeddings(self, embedding1, embedding2):
        """Process merges 2 embeddings into one."""
        pass
