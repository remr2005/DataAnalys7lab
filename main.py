import csv
from dsmltf import tokenize, count_words, spam_probability
import nltk
from nltk import pos_tag
from collections import defaultdict

def make_data() -> list:
    # Загружаем необходимые ресурсы
    nltk.download("averaged_perceptron_tagger_eng")
    # Парсим данные
    with open("spam.csv") as f:
        data = []
        for i in csv.reader(f):
            if i[0]=="spam": data.append([i[1],1])
            else: data.append([i[1],0])
    return data
    
def main() -> None:
    dataset = make_data()
    # количество спамных сообщений
    spam_count = len([i for i in dataset if i[1]])
    ham_count = len(dataset) - spam_count
    # тренировочные данные
    train_set = count_words(dataset[:-100])
    # словарь с определением части речи  слова
    tagged_keys = pos_tag(train_set.keys())
    # отфильтрованные данные(без прилогательных)
    train_set = defaultdict(lambda: [0, 0], {key: train_set[key] for key, tag in tagged_keys if tag not in ('JJ', 'JJR', 'JJS')})
    # самые часто встречающиееся слова в спаме
    words = sorted(train_set.keys(), key=lambda x: train_set[x][0] if len(x) >= 5 else 0)[-7:]
    words = [(i, train_set[i][0]/spam_count, train_set[i][1]/ham_count)  for i in words]
    print(words)

if __name__ == "__main__":
    main()
