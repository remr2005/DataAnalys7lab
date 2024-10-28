import csv
from dsmltf import tokenize, count_words, spam_probability, f1_score
import nltk
from nltk import pos_tag
from collections import defaultdict

def word_probabilities(counts:list[tuple], total_spams:int, total_non_spams:int, k=0.5) -> list[tuple]:
    return [(w[0], (w[1] + k) / (total_spams + 2*k),
             (w[2] + k) / (total_non_spams + 2*k))
            for w in counts]

def make_data() -> list:
    # Загружаем необходимые ресурсы
    nltk.download("averaged_perceptron_tagger_eng")
    # Парсим данные
    with open("spam.csv") as f:
        data = []
        for i in csv.reader(f): data.append([i[1], 1 if i[0]=="spam" else 0])
    return data
    
def test(words: list[tuple], dataset: list) -> float:
    true_pos, false_pos, false_neg = 0, 0, 0
    for i in dataset:
        match round(spam_probability(words, i[0])),i[1]:
            case 1,1:true_pos+=1
            case 1,0:false_pos+=1
            case 0,1:false_neg+=1
    return f1_score(true_pos, false_pos, false_neg)

def main() -> None:
    dataset = make_data()
    # количество спамных сообщений
    spam_count = len([i for i in dataset if i[1]])
    ham_count = len(dataset) - spam_count
    # тренировочные данные
    train_set = count_words(dataset[:-35])
    # словарь с определением части речи  слова
    tagged_keys = pos_tag(train_set.keys())
    # отфильтрованные данные(без прилогательных)
    train_set = defaultdict(lambda: [0, 0], {key: train_set[key] for key, tag in tagged_keys if tag not in ('JJ', 'JJR', 'JJS')})
    # самые часто встречающиееся слова в спаме
    words = sorted(train_set, key=lambda x: train_set[x][0] if len(x) >= 5 else 0)[-7:]
    words = [(i, train_set[i][0]/spam_count, train_set[i][1]/ham_count if train_set[i][1]/ham_count else 0.01)  for i in words]
    # пробуем без сглаживания
    print("", test(words, dataset[-35:]))
    # проведем сглаживание
    words = word_probabilities([(i[0], train_set[i[0]][0], train_set[i[0]][1]) for i in words], spam_count, ham_count)
    print(test(words, dataset[-35:]))

if __name__ == "__main__":
    main()
