import json

fileName = 'TaleantQA.txt'

outputJson = {}

intents = []

allPatterns = []
allResponses = []
allTags = []

def readEachLine(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line

def generateKB(fileName):

    with open(fileName,encoding="utf8") as f_in:
        for line in readEachLine(f_in):
            marker = line.split()[0]
        

            if marker=="#":

                patterns = line.split("# ")[1:]
                allPatterns.append(patterns)

            elif marker == "##":

                tag = line.split("## ")[1]
                allTags.append(tag)

            elif marker == "###":

                responses = line.split("### ")[1:]
                allResponses.append(responses)
    #print(len(allTags))
    for pattern, response, tag in zip(allPatterns, allResponses, allTags):
        outputIntent = {}
        outputIntent["tag"] = tag
        outputIntent["patterns"] = pattern
        outputIntent["responses"] = response
        outputIntent["context_set"] = ""
        intents.append(outputIntent)

    outputJson["intents"] = intents


    with open('intentsNew.json', 'w') as outfile:
        json.dump(outputJson, outfile,indent=4)

    # To make things easy for now, will comment out later
    with open('C:\\Users\\Khaleda\\Desktop\\NAFIS\\chatbot\\API\\ChatbotV4_Flask_SimpleWebInterface\\intents.json', 'w') as outfile2:
        json.dump(outputJson, outfile2,indent=4)

    with open('C:\\Users\\Khaleda\\Desktop\\NAFIS\\chatbot\\API\\ChatbotV4_DjangoRest\\data\\taleant\\intents.json', 'w') as outfile3:
        json.dump(outputJson, outfile3,indent=4)

generateKB(fileName)