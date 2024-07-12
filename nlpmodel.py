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

def process_data(lang1, lang2):
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

    return input_lang, output_lang, pairs

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
def trainModel(model, input_lang, output_lang, pairs, num_iteration=20000):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(num_iteration)]

    for iter in range(1, num_iteration+1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = Model(model, input_tensor, target_tensor, optimizer, criterion) #model객체를 이용하여 오차 계산
        total_loss_iterations += loss

        if iter % 5000 == 0:
            average_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print('%d %.4f' % (iter, average_loss))

    torch.save(model.state_dict(), './min0/mytraining.pt')
    return model


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

#모델 훈련#
lang1 = 'eng' #입력으로 사용할 영어
lang2 = 'fra' #출력으로 사용할 프랑스어
input_lang, output_lang, pairs = process_data(lang1, lang2)

randomize = random.choice(pairs)
print('random sentence {}'.format(randomize))

input_size = input_lang.n_words
output_size = output_lang.n_words
print('Input : {} Output : {}'.format(input_size, output_size)) #입력과 출력에 대한 단어 수 출력

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 5000 #75000

encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)
#디코더의 첫 번째 입력으로 <SOS>토큰이 제공되고, 인코더의 마지막 은닉 상태가 디코더의 첫 번째 은닉 상태로 제공된다.

model = Seq2Seq(encoder, decoder, device).to(device) #인코더-디코더 모델(seq2seq)의 객체 생성

print(encoder)
print(decoder)

model = trainModel(model, input_lang, output_lang, pairs, num_iteration) #모델 학습

print(evaluateRandomly(model, input_lang, output_lang, pairs))
