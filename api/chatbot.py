import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random
import json
import filecmp
from shutil import copyfile
import os.path
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
import aiml
import os

stemmer = LancasterStemmer()

main_current_kb_file = 'intents.json'
main_previous_kb_file_copy = 'intentsPreviousCopy.json'
stop_words_file = 'stop_words.txt'


def check_kb_for_change(current_kb_file, previous_kb_file_copy):
    if not filecmp.cmp(current_kb_file, previous_kb_file_copy):
        try:
            copyfile(current_kb_file, previous_kb_file_copy)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)
        return True
    else:
        return False


def load_data(current_kb_file):
    with open(current_kb_file) as file:
        data = json.load(file)
    return data


def load_stop_words(stop_words_file):
    stop_words = []

    file = open(stop_words_file)

    for word in file.read().split():
        stop_words.append(word)

    return stop_words


def match_token_with_vocab(vocabulary, token_list):
    stemmed_tokens = [stemmer.stem(w.lower()) for w in token_list if w != "?"]

    result = any(x in vocabulary for x in stemmed_tokens)

    return result


def get_total_vocabulary(data):
    vocabulary = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            words = nltk.word_tokenize(pattern)
            vocabulary.extend(words)

        for response in intent["responses"]:
            words = nltk.word_tokenize(response)
            vocabulary.extend(words)

    vocabulary = [stemmer.stem(w.lower()) for w in vocabulary if w != "?"]
    vocabulary = sorted(list(set(vocabulary)))
    return vocabulary


def remove_stop_words(param, words_to_remove):
    result = [i for i in param if i not in words_to_remove]

    return result


def preprocess_data(kb_file_updated, data):
    words = []
    labels = []
    training = []
    output = []

    if kb_file_updated == False and os.path.exists("data.pickle"):
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    else:
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(list(set(labels)))

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    return words, labels, training, output


def build_model(kb_file_updated, training, output):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(output[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if not kb_file_updated:
        try:
            print("in try?")
            with open(f'chatbotV4_model.pkl', 'rb') as f:
                model = pickle.load(f)
        except:
            print("in except?")
            model.fit(np.array(training), np.array(output), epochs=5000, batch_size=10, verbose=1)
            pickle.dump(model, open("chatbotV4_model.pkl", "wb"))
    else:
        print("in else?")
        model.fit(np.array(training), np.array(output), epochs=5000, batch_size=10, verbose=1)
        pickle.dump(model, open("chatbotV4_model.pkl", "wb"))

    return model


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


KBFileUpdated = check_kb_for_change(main_current_kb_file, main_previous_kb_file_copy)

data = load_data(main_current_kb_file)
stops = load_stop_words(stop_words_file)

words, labels, training, output = preprocess_data(KBFileUpdated, data)
model = build_model(KBFileUpdated, training, output)
vocab = get_total_vocabulary(data)

graph = tf.get_default_graph()
confidenceThreshold = 0.85
confidenceThresholdMedium = 0.60
confidenceThresholdLow = 0.30


def initialize_ai_ml_model():
    AIML_MODEL = "aimlModel.dump"

    ai_ml_kernel = aiml.Kernel()

    # This code checks if a dump exists and
    # otherwise loads the aiml from the xml files
    # and saves the brain dump.
    if os.path.exists(AIML_MODEL):
        print("Loading from aimlModel file: " + AIML_MODEL)
        ai_ml_kernel.loadBrain(AIML_MODEL)
    else:
        print("Parsing aiml files")
        ai_ml_kernel.bootstrap(learnFiles="std-startup.aiml", commands="load aiml b")
        print("Saving aimlModel file: " + AIML_MODEL)
        ai_ml_kernel.saveBrain(AIML_MODEL)

    return ai_ml_kernel


def respond_using_ai_ml(input, aiml_kernel):
    response = aiml_kernel.respond(input)

    returning_object = JsonResponse(response, safe=False)
    print("Status: " + str(returning_object.status_code))
    print(response)
    return returning_object


aimlKernel = initialize_ai_ml_model()
USE_AIML = True


def respond_using_kb(tag):
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    result_response = str(random.choice(responses))

    returning_object = JsonResponse(result_response, safe=False)
    print("Status: " + str(returning_object.status_code))
    print(result_response)
    return returning_object


def respond_with_sorry():
    message = "Sorry, I don't understand. Try again with another question, or you can forward your enquiry towards our <a href=\"https://www.taleant.com/enquiry\" target=\"_blank\">Support Team</a>"
    returning_object = JsonResponse(message, safe=False)
    print("Status: " + str(returning_object.status_code))
    print(message)
    return returning_object


@api_view(['GET', 'POST'])
def chat(query):
    while True:
        try:
            temporary_input_message = query.POST['messageText'].encode('utf-8').strip()

            input_message = (temporary_input_message.decode('utf-8')).lower()
            print("Query: " + input_message)

            if input_message == "quit":
                exit(1)

            ##########################################################################
            input_tokens_with_stop_words = nltk.word_tokenize(input_message)
            input_tokens_with_out_stops_words = remove_stop_words(input_tokens_with_stop_words, stops)
            print("input with stop words:", input_tokens_with_stop_words)
            print("input without stop words:", input_tokens_with_out_stops_words)

            match_result_with_stops = match_token_with_vocab(vocab, input_tokens_with_stop_words)
            match_result_without_stops = match_token_with_vocab(vocab, input_tokens_with_out_stops_words)

            print("With stops: ", match_result_with_stops)
            print("Without stops: ", match_result_without_stops)
            ##########################################################################

            input_data = pd.DataFrame([bag_of_words(input_message, words)], dtype=float, index=['input'])

            with graph.as_default():

                results = model.predict([input_data])[0]

            results_index = np.argmax(results)
            tag = labels[results_index]

            if results[results_index] > confidenceThreshold and match_result_with_stops == True:
                print("inside the if??")
                print("My Confidence: ", results[results_index], "Tag: ", tag)
                return respond_using_kb(tag)

            elif results[results_index] >= confidenceThresholdMedium and match_result_without_stops == True:
                print("inside the ELIF??")
                print("My Confidence: ", results[results_index], "Tag: ", tag)
                return respond_using_kb(tag)

            else:
                if USE_AIML:
                    if results[results_index] >= confidenceThresholdLow and match_result_with_stops == True:
                        print("inside aiml")
                        print("My Confidence: ", results[results_index], "Tag: ", tag)
                        return respond_using_ai_ml(input_message, aimlKernel)
                    else:
                        print("inside aiml sorry")
                        print("My Confidence: ", results[results_index], "Tag: ", tag)
                        return respond_with_sorry()

                else:
                    print("inside just sorry")
                    print("My Confidence: ", results[results_index], "Tag: ", tag)
                    return respond_with_sorry()

        except ValueError as e:
            return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

# chat()
