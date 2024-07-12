import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import re
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data ready#
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#data normalization#
def normalizeString(df, lang):
    sentence = df[lang].str.lower() #소문자 변환
    sentence = sentence.str.replace('[^A-Za-z\s]+', '') #알파벳 제외 공백화
    sentence = sentence.str.normalize('NFD') #유니코드 정규화 방식
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence

def read_sentence(df, lang1, lang2):
    sentence1 = normalizeString(df, lang1) #데이터셋의 첫번째 열(영어)
    sentence2 = normalizeString(df, lang2) #데이터셋으 두번째 열(프랑스어)
    return sentence1, sentence2

def read_file(loc, lang1, lang2):
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2])
    return df

def process_data(lang1, lang2, train_ratio=0.8):
    df = read_file('./min0/%s-%s.txt' % (lang1, lang2), lang1, lang2)
    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    input_lang = Lang()
    output_lang = Lang()
    pairs = []
    for i in range(len(df)):
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            full = [sentence1[i], sentence2[i]] #첫번째와 두번째 열을 합쳐서 저장
            input_lang.addSentence(sentence1[i]) #입력(input)으로 영어를 사용
            output_lang.addSentence(sentence2[i]) #출력(output)으로 프랑스어 사용
            pairs.append(full)

    # 데이터를 나누는 비율을 기준으로 훈련용과 테스트용으로 나눔
    random.shuffle(pairs)  # 데이터를 섞음
    split_idx = int(len(pairs) * train_ratio)  # 데이터를 나누는 인덱스 계산

    train_pairs = pairs[:split_idx]  # 훈련용 데이터셋
    test_pairs = pairs[split_idx:]  # 테스트용 데이터셋

    return input_lang, output_lang, train_pairs, test_pairs

#텐서로 변환(데이터 쌍pair를 텐서로 변환해야함)
#데이터셋이 문장이기 때문에 문장의 모든 끝에 입력이 완료되었음을
#네트워크에 알려줘야 하는데 이를 토큰이라고 함

#문장을 단어로 분리하고 인덱스를 반환
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

#딕셔너리에서 단어에 대한 인덱스를 가져오고 문장 끝에 토큰 추가
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

#입력과 출력 문장을 텐서로 변환하여 반환
def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return(input_tensor, target_tensor)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim #인코더에서 사용할 입력층
        self.embbed_dim = embbed_dim #인코더에서 사용할 임베딩 계층
        self.hidden_dim = hidden_dim #인코더에서 사용할 은닉층(이전 은닉층)
        self.num_layers = num_layers #인코더에서 사용할 GRU의 계층 수
        self.embedding = nn.Embedding(input_dim, self.embbed_dim) #임베딩 계층 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers) #임베딩 차원, 은닉층 차원, GRU의 계층 개수를 이용하여 GRU 계층을 초기화

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1) #임베딩 처리
        outputs, hidden = self.gru(embedded) #임베딩 결과를 GRU모델에 적용
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, self.embbed_dim) #임베딩 계층 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers) #GRU 계층 초기화
        self.out = nn.Linear(self.hidden_dim, output_dim) #선형 계층 초기화
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1,-1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()

        self.encoder = encoder #인코더 초기화
        self.decoder = decoder #디코더 초기화
        self.device = device

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):
        input_length = input_lang.size(0) #입력 문자 길이(문장의 단어 수)
        batch_size = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device) #예측된 출력을 저장하기 위한 변수 초기화

        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_lang[i]) #문장의 모든 단어 인코딩
        decoder_hidden = encoder_hidden.to(device) #인코더의 은닉층을 디코더의 은닉층으로 사용
        decoder_input = torch.tensor([SOS_token], device=device) #첫 번째 예측 단어 앞에 토큰(SOS) 추가

        for t in range(target_length): #현재 단어에서 출력 단어를 예측
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            #티처포스는 seq2seq 모델에서 많이 사용되는 기법이다.
            #티처포스는 번역(예측)하려는 목표 단어를 디코더의 다음 입력으로 넣어주는 기법
            #장점 : 초기 안정적인 학습
            #단점 : 네트워크가 불안정해질 수 있음
            topv, topi = decoder_output.topk(1)
            input = (output_lang[t] if teacher_force else topi) #teacher_force를 활성화하면 목표 단어를 다음 입력으로 사용
            if(teacher_force == False and input.item() == EOS_token): #teacher_force를 활성화하지 않으면 자체 예측 값을 다음 입력으로 사용
                break
        return outputs

#모델의 오차 계산 함수 정의#
teacher_forcing_ratio = 0.5

def Model(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])
    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss

#모델 훈련 함수 정의#
# def trainModel(model, input_lang, output_lang, train_pairs, num_iteration=20000):
def trainModel(model, input_lang, output_lang, train_pairs, idxs=100, num_iteration=2):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0
    epoch_loss = 0

    # training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(train_pairs)) for i in range(num_iteration)]
    training_pairs = [tensorsFromPair(input_lang, output_lang, train_pairs[i]) for i in idxs]

    for iter in range(1, num_iteration+1):
        # training_pair = training_pairs[iter-1]
        training_pair = random.choice(training_pairs, replace=False)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        epoch_loss = Model(model, input_tensor, target_tensor, optimizer, criterion) #model객체를 이용하여 오차 계산
        total_loss_iterations += epoch_loss

        if iter % 10 == 0:
            average_loss = total_loss_iterations / 10
            total_loss_iterations = 0
            print('%d %.4f' % (iter, average_loss))

    torch.save(model.state_dict(), './min0/mytraining.pt')
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


#모델 평가#
def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0]) #입력 문자열을 텐서로 변환
        output_tensor = tensorFromSentence(output_lang, sentences[1]) #출력 문자열을 텐서로 변환
        decoded_words = []
        
        output = model(input_tensor, output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1) #각 출력에서 가장 높은 값을 찾아 인덱스를 반환

            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>') #EOS 토큰을 만나면 평가를 멈춘다.
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])
        return decoded_words

def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs) #임의로 문장을 가져옴
        print('input {}'.format(pair[0]))
        print('output {}'.format(pair[1]))
        output_words = evaluate(model, input_lang, output_lang, pair) #모델 평가 결과는 output_words에 저장
        output_sentence = ' '.join(output_words)
        print('predicted {}'.format(output_sentence))




# iid and non-iid
def read_text_file(file_path):
    """
    Read text data from a single text file with license information
    :param file_path: path to the text file
    :return: list of sentences with license information
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

def preprocess_text_data(data):
    """
    Preprocess text data with license information (e.g., tokenization, lowercasing)
    :param data: list of sentences with license information
    :return: preprocessed data
    """
    # Perform any necessary preprocessing steps here
    preprocessed_data = [line.split('\t')[0].strip() for line in data]
    return preprocessed_data



def iid_nlp(file_path, num_users):
    """
    Sample I.I.D. client data from a single text file for NLP tasks
    :param file_path: path to the text file
    :param num_users: number of clients
    :return: dict of sentence index
    """
    # Read and preprocess text data
    data = read_text_file(file_path)
    preprocessed_data = preprocess_text_data(data)

    num_items = int(len(preprocessed_data) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(preprocessed_data))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def noniid_nlp(file_path, num_users):
    """
    Sample non-I.I.D client data from a single text file for NLP tasks
    -> Different clients can hold vastly different amounts of data
    :param file_path: path to the text file
    :param num_users: number of clients
    :return: dict of sentence index
    """
    # Read and preprocess text data
    data = read_text_file(file_path)
    preprocessed_data = preprocess_text_data(data)

    num_dataset = len(preprocessed_data)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(num_users)}

    min_num = 100
    max_num = 700

    random_num_size = np.random.randint(min_num, max_num + 1, size=num_users)

    assert num_dataset >= sum(random_num_size)

    for i, rand_num in enumerate(random_num_size):
        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set

    return dict_users


# class LocalUpdate(object):
#     def __init__(self, args, dataset=None, idxs=None):
#         self.args = args
#         self.loss_func = nn.CrossEntropyLoss()
#         self.selected_clients = []
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, \
#                                     num_workers=2, persistent_workers=True)
#
#     def train(self, net):
#         net.train()
#         # train and update
#         optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
#
#         epoch_loss = []
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train):
#                 images, labels = images.to(self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 # print("label is :", labels, len(labels))
#                 loss = self.loss_func(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 if self.args.verbose and batch_idx % 10 == 0:
#                     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         iter, batch_idx * len(images), len(self.ldr_train.dataset),
#                               100. * batch_idx / len(self.ldr_train), loss.item()))
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss) / len(batch_loss))
#
#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

#모델 훈련#
lang1 = 'eng' #입력으로 사용할 영어
lang2 = 'fra' #출력으로 사용할 프랑스어
input_lang, output_lang, train_pairs, test_pairs = process_data(lang1, lang2)

# m = max(int(args.frac * args.num_users), 1)
# f = int(args.frac * args.num_comps)
#stub code
m = 10
f = 1

dict_users = noniid_nlp('./min0/eng-fra.txt', m)

# randomize = random.choice(pairs)
# print('random sentence {}'.format(randomize))

input_size = input_lang.n_words
output_size = output_lang.n_words
print('Input : {} Output : {}'.format(input_size, output_size)) #입력과 출력에 대한 단어 수 출력

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 100 #75000

encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)
#디코더의 첫 번째 입력으로 <SOS>토큰이 제공되고, 인코더의 마지막 은닉 상태가 디코더의 첫 번째 은닉 상태로 제공된다.

# model = Seq2Seq(encoder, decoder, device).to(device) #인코더-디코더 모델(seq2seq)의 객체 생성
net_glob = Seq2Seq(encoder, decoder, device).to(device) #인코더-디코더 모델(seq2seq)의 객체 생성

print(encoder)
print(decoder)

# copy weights
w_glob = net_glob.state_dict()
w_glob_local = net_glob.state_dict()

# training
attack_list = []
loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []

acc_train = [];loss_train = [];acc_test = [];loss_test = []

all_clients = False #args.all_client 전체 훈련시킬건지 여부
num_users = 100 #args.num_users
if all_clients:
    print("Aggregation over all clients")
    w_locals = [w_glob for i in range(num_users)]
    loss_locals = [0. for i in range(num_users)]
else:
    w_locals = [w_glob for i in range(m)]
    loss_locals = [0. for i in range(m)]

loss_avg = 0.0
for iter in range(start_iters, args.epochs):

    print('\033[36m' + "{} rnd start!".format(iter) + '\033[0m')
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    print("idxs_users is : ", idxs_users)

    for idx in range(len(idxs_users)): #100명중 m명 진행
        ### Benign training
        w, loss = trainModel(net_glob, input_lang, output_lang, dict_users, num_iteration)  # 모델 학습
        # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idxs_users[idx]])
        # w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        w_locals[idx] = copy.deepcopy(w)
        loss_locals[idx] = copy.deepcopy(loss)


    # update global weights
    print("Global aggregation start!")
    num_users = m;
    num_comps = f;
    print("num_users and num_comps :", num_users, num_comps)


    w_glob, losses = FedAvg(w_locals, loss_locals)

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

model = trainModel(netglob, input_lang, output_lang, pairs, num_iteration) #모델 학습

###########################################################################################
# #어텐션이 적용된 디코더#
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size) #임베딩 계층 초기화
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         #어텐션은 입력을 디코더로 변환한다. 즉, 어텐션은 입력 시퀀스와 길이가 같은 인코딩된 시퀀스로 변환하는 역할을 한다.
#         #따라서 self.max_length는 모든 입력 시퀀스의 최대 길이어야 한다.
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1,1,-1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]),1)), dim = 1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
#         #torch.bmm 함수는 배치 행렬 곱(BMM)을 수행하는 함수로 두 개 이상의 차원을 지닌 텐서가 주어졌을 때 뒤의 두 개 차원에 대해 행렬 곱을 수행하는 함수이다.
#         #즉, 가중치와 인코더의 출력 벡터를 곱하겠다는 의미이며, 그 결과(attn_applied)는 입력 시퀀스의 특정 부분에 관한 정보를 포함하고 있기 때문에 디코더가 적절한 출력 단어를 선택하도록 도와준다.
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         output = F.relu(output)
#         output, hidden = self.gru(outpudatat, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
# def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0
#     plot_loss_total = 0
#
#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#     training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(n_iters)]
#     criterion = nn.NLLLoss()
#
#     for iter in range(1, n_iters+1):
#         training_pair = training_pairs[iter - 1]
#         input_tensor = training_pair[0] #입력 + 출력 쌍에서 입력을 input_tensor로 사용
#         target_tensor = training_pair[1] # 입력 + 출력 쌍에서 출력을 target_tensor로 사용
#         loss = Model(model, input_tensor, target_tensor, decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss
#
#         if iter % 5000 == 0:
#             print_loss_avg = print_loss_total / 5000
#             print_loss_total = 0
#             print('%d, %.4f' % (iter, print_loss_avg))
#
# #어텐션 디코더 모델 훈련#
# import time
# embed_size = 256
# hidden_size = 512
# num_layers = 1
# input_size = input_lang.n_words
# output_size = output_lang.n_words
#
# encoder1 = Encoder(input_size, hidden_size, embed_size, num_layers)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)
#
# print(encoder1)
# print(attn_decoder1)
#
# attn_model = trainIters(encoder1, attn_decoder1, 75000, print_every=5000, plot_every=100, learning_rate=0.01)
###########################################################################################



