import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random

import json

import filecmp
from shutil import copyfile
from sys import exit
import os.path

from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings

from rest_framework.renderers import JSONRenderer

import aiml
import os

from taleant.helperFile import supportURL

# Model manipulation
##########################
from taleant.models import QueryPatterns

# FOR API AUTHENTICATION
########################################
from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authtoken.models import Token
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_200_OK
)

from rest_framework.permissions import IsAuthenticated


@csrf_exempt
@api_view(["POST"])
@permission_classes((AllowAny,))
def login(request):
    username = request.data.get("username")
    password = request.data.get("password")

    if username is None or password is None:
        return Response({'error': 'Please provide both username and password'},
                        status=HTTP_400_BAD_REQUEST)
    user = authenticate(username=username, password=password)
    if not user:
        return Response({'error': 'Invalid Credentials'},
                        status=HTTP_404_NOT_FOUND)
    token, _ = Token.objects.get_or_create(user=user)
    return Response({'token': token.key},
                    status=HTTP_200_OK)


########################################


# currentKBFile = 'data\\taleant\\intents.json'
# previousKBFileCopy = 'data\\taleant\\intentsPreviousCopy.json'
# stopWordsFile = 'data\\taleant\\stop_words.txt'
# dataPickleFile = 'data\\taleant\\data.pickle'
# mlModelFile = 'data\\taleant\\ml_model.pkl'

currentKBFile = os.path.join('data/taleant/intents.json')
previousKBFileCopy = os.path.join('data/taleant/intentsPreviousCopy.json')
stopWordsFile = os.path.join('data/taleant/stop_words.txt')
dataPickleFile = os.path.join('data/taleant/data.pickle')
mlModelFile = os.path.join('data/taleant/ml_model.pkl')


def checkKBForChange(currentKBFile, previousKBFileCopy):
    if (not filecmp.cmp(currentKBFile, previousKBFileCopy)):
        try:
            copyfile(currentKBFile, previousKBFileCopy)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)
        return True
    else:
        return False


def loadData(currentKBFile):
    with open(currentKBFile) as file:
        data = json.load(file)
    return data


# FOR AUTO SUGGESTION FEATURE
#############################
def insertQueryPatternsIntoDB(data):
    QueryPatterns.objects.all().delete()
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            q = QueryPatterns()
            q.query = pattern
            q.save()


def getQueryPatternFromDB():
    print("PRINTING FROM DB")
    dbContents = QueryPatterns.objects.all()

    for q in dbContents:
        print(q)


##############################

def loadStopWords(stopWordsFile):
    stopWords = []

    file = open(stopWordsFile)

    for word in file.read().split():
        stopWords.append(word)

    return stopWords


def matchTokenWithVocab(vocab, tokenList):
    stemmedTokens = [stemmer.stem(w.lower()) for w in tokenList if w != "?"]

    result = any(x in vocab for x in stemmedTokens)

    return result


def getTotalVocab(data):
    vocab = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            vocab.extend(wrds)
        for response in intent["responses"]:
            wrds = nltk.word_tokenize(response)
            vocab.extend(wrds)

    vocab = [stemmer.stem(w.lower()) for w in vocab if w != "?"]
    vocab = sorted(list(set(vocab)))
    return vocab


def removeStopWords(intput, wordsToRemove):
    result = [i for i in intput if i not in wordsToRemove]

    return result


def preprocessData(KBFileUpdated, data):
    words = []
    labels = []
    training = []
    output = []

    if (KBFileUpdated == False and os.path.exists(dataPickleFile)):
        with open(dataPickleFile, "rb") as f:
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

        # FOR AUTO SUGGESTION FEATURE
        ##############################
        # insertQueryPatternsIntoDB(data)
        ##############################

        with open(dataPickleFile, "wb") as f:
            pickle.dump((words, labels, training, output), f)

    return words, labels, training, output


def buildModel(KBFileUpdated, training, output):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(output[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if (KBFileUpdated == False):
        try:
            print("in try?")
            with open(mlModelFile,
                      'rb') as f:  # was like f'chatbotV4_model.pkl', keep it here for backup error resolving
                model = pickle.load(f)
        except:
            print("in except?")
            model.fit(np.array(training), np.array(output), epochs=5000, batch_size=10, verbose=1)
            pickle.dump(model, open(mlModelFile, "wb"))
    else:
        print("in else?")
        model.fit(np.array(training), np.array(output), epochs=5000, batch_size=10, verbose=1)
        pickle.dump(model, open(mlModelFile, "wb"))

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


KBFileUpdated = checkKBForChange(currentKBFile, previousKBFileCopy)

data = loadData(currentKBFile)
stops = loadStopWords(stopWordsFile)

tagsForSupport = ["bot dumb", "cost business user", "register", "delete profile", "link expired", "trouble applying",
                  "email forget", "help after ad"]

words, labels, training, output = preprocessData(KBFileUpdated, data)
model = buildModel(KBFileUpdated, training, output)
vocab = getTotalVocab(data)

graph = tf.get_default_graph()
confidenceThreshold = 0.85
confidenceThresholdMedium = 0.50
confidenceThresholdLow = 0.30


def initializeAIMLModel():
    AIML_MODEL = "aimlModel.dump"

    aimlKernel = aiml.Kernel()

    # This code checks if a dump exists and
    # otherwise loads the aiml from the xml files
    # and saves the brain dump.
    if os.path.exists(AIML_MODEL):
        print("Loading from aimlModel file: " + AIML_MODEL)
        aimlKernel.loadBrain(AIML_MODEL)
    else:
        print("Parsing aiml files")
        aimlKernel.bootstrap(learnFiles="std-startup.aiml", commands="load aiml b")
        print("Saving aimlModel file: " + AIML_MODEL)
        aimlKernel.saveBrain(AIML_MODEL)

    return aimlKernel


def respondUsingAIML(input, aimlKernel):
    response = aimlKernel.respond(input)

    jsonContent = {"response": response, "status": "OK"}

    returningObject = JsonResponse(jsonContent, safe=False)
    print("Status: " + str(returningObject.status_code))
    print(jsonContent)
    return returningObject


aimlKernel = initializeAIMLModel()
USE_AIML = False


def respondUsingKB(tag):
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    resultResponse = str(random.choice(responses))

    if tag in tagsForSupport:
        jsonContent = {"status": "OK", "response": resultResponse, "url": supportURL}

        returningObject = JsonResponse(jsonContent, safe=False)
        print("Status: " + str(returningObject.status_code))
        print(jsonContent)
        return returningObject
    else:
        jsonContent = {"status": "OK", "response": resultResponse}

        returningObject = JsonResponse(jsonContent, safe=False)
        print("Status: " + str(returningObject.status_code))
        print(jsonContent)
        return returningObject


def respondWithSorry():
    # sorryMessage = "Sorry, I don't understand. Try again with another question, or you can forward your enquiry towards our <a href=\"https://www.taleant.com/enquiry\" target=\"_blank\">Support Team</a>"
    sorryMessage = "Sorry, I don't understand. Try again with another question, or you can forward your enquiry towards our Support Team"

    jsonContent = {"status": "OK", "response": sorryMessage, "url": supportURL}

    returningObject = JsonResponse(jsonContent, safe=False)
    print("Status: " + str(returningObject.status_code))
    print(jsonContent)
    return returningObject


# FOR AUTO SUGGESTION FEATURE
##############################
# getQueryPatternFromDB()
##############################

@csrf_exempt
@api_view(['GET', 'POST'])
def chat(query):
    try:
        temporaryInputMessage = query.POST['messageText'].encode('utf-8').strip()

        inputMessage = (temporaryInputMessage.decode('utf-8')).lower()
        print("Query: " + inputMessage)

        if inputMessage == "quit":
            exit(1)

        ##########################################################################
        inputTokensWithStopWords = nltk.word_tokenize(inputMessage)
        inputTokensWithOutStopsWords = removeStopWords(inputTokensWithStopWords, stops)
        print("input with stop words:", inputTokensWithStopWords)
        print("input without stop words:", inputTokensWithOutStopsWords)

        matchResultWithStops = matchTokenWithVocab(vocab, inputTokensWithStopWords)
        matchResultWithoutStops = matchTokenWithVocab(vocab, inputTokensWithOutStopsWords)

        print("With stops: ", matchResultWithStops)
        print("Without stops: ", matchResultWithoutStops)
        ##########################################################################

        input_data = pd.DataFrame([bag_of_words(inputMessage, words)], dtype=float, index=['input'])

        with graph.as_default():

            results = model.predict([input_data])[0]

        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > confidenceThreshold and matchResultWithStops == True:
            print("inside the if??")
            print("My Confidence: ", results[results_index], "Tag: ", tag)
            return respondUsingKB(tag)

        elif results[results_index] >= confidenceThresholdMedium and matchResultWithoutStops == True:
            print("inside the ELIF??")
            print("My Confidence: ", results[results_index], "Tag: ", tag)
            return respondUsingKB(tag)

        else:
            if (USE_AIML == True):
                if results[results_index] >= confidenceThresholdLow and matchResultWithStops == True:
                    print("inside aiml")
                    print("My Confidence: ", results[results_index], "Tag: ", tag)
                    return respondUsingAIML(inputMessage, aimlKernel)
                else:
                    print("inside aiml sorry")
                    print("My Confidence: ", results[results_index], "Tag: ", tag)
                    return respondWithSorry()

            else:
                print("inside just sorry")
                print("My Confidence: ", results[results_index], "Tag: ", tag)
                return respondWithSorry()

    except:
        jsonContent = {"status": "EXCEPTION", "response": "NULL", "url": supportURL}
        returningObject = JsonResponse(jsonContent, safe=False)

        return returningObject
