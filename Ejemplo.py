import nltk
import gensim
nltk.download('punkt')

# Cargar el documento
with open('reglamento_transito.txt', 'r', encoding='utf-8') as file:
    document = file.read()

# Tokenizar el documento en oraciones
sentences = nltk.sent_tokenize(document)

# Tokenizar cada oración en palabras
word_tokens = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

# Entrenar el modelo de Word2Vec
modelo_w2v = gensim.models.Word2Vec(sentences=word_tokens,
                                    vector_size=50,
                                    epochs=100,  # Cambiado 'iter' a 'epochs'
                                    min_count=1)

# Obtener las 10 palabras más similares
similar_words = modelo_w2v.wv.most_similar('auto')
print("Palabras más similares a 'auto':", similar_words)

# Vector de la palabra 'auto'
vector_auto = modelo_w2v.wv['auto']
print("Vector de 'auto':", vector_auto)
