import json
import re
import random

def makeGraph():
    with open('appSpeechData.json') as json_file:
        information = json.load(json_file)

    print (information)
    ##print(information)
    ##print(information.keys())
    ##print(information.values())
    #
    #k = []
    #for key in information.keys():
    #    k.append(key)
    #
    #v = []
    #for val in information.values():
    #    v.append(float(val))
    #
    #fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
    #                     cells=dict(values=[k, v]))
    #                                          ])
    #fig.write_image("graphs/audioTable.jpg")
    #
    #fig.show(render="jpg")
    #
    with open("appFaceData.json") as face_file:
        faceData = json.load(face_file)
    
    ##print(faceData)
    
    questionAverages = dict()  ## its very important that the question folders are just named 1, 2, 3...
    i = 1
    j = 1
    flatData = dict()  ## this is all of the frames from all questions
    score = 0
    score_counter = 0
    for K, V in faceData.items():
        print(K + " " + str(V))
        questionNum = re.sub("\D", "", K)
        print(questionNum)
        audioOffSetWeight = information[questionNum]
        print(audioOffSetWeight)
        ans = 0
        for k, v in V.items():
            ans = ans + V[k]
            V[k] = 0.3*V[k] + 0.7*audioOffSetWeight
            ##print(V[k])
            it = 0
            score_counter = score_counter+1
            score = score +V[k]
            while it < 10:  ## This pads each real data point with 10 random offset ones
                flatData[j] = V[k] + random.random() / 10
                j = j+1
                it = it +1
        ans = ans / len(V)
        questionAverages[i] = ans
        i = i+1
    
    score = score / score_counter
    
    ##print(questionAverages)
    ##print(faceData)
    ##print(flatData)
    ## print("AVERAGE")
    print("Raw Score: " + str(score))
    verdict = "Emotionally Distressed"
    if(score > -0.15):
        verdict = "Healthy"
    if(score > 2.5):
        verdict = "Overjoyed"
    print("Verdict: " + str(verdict))
    
    f = open("results","w+")
    f.write(str(score)+ '\n')
    f.write(verdict)
    f.close()
    
    ## By the end facedata is updated to be the 
        ## weighted score between face data and audio data
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    D = flatData
    
    plt.plot(range(len(D)), list(D.values()))
    plt.ylabel('Emotion Score')
    plt.xlabel('Time (frames)')
    plt.savefig("mastergraph.png")
    plt.show()
    
    labels = ["Question", "Emotion Score"]
    pictureRows = [ [str(k),round(v, 3)] for k, v in questionAverages.items() ]
    
    
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    table = ax.table(cellText=pictureRows, colLabels = labels,loc='center')
    table.set_fontsize(20)
    table.scale(1,3)
    ax.axis('off')
    
    plt.savefig("pictureChart.png")
    plt.show()
    
    
    
    audioRows = [ [str(k),round(v, 3)] for k, v in information.items() ]
    
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    table = ax.table(cellText=audioRows, colLabels = labels,loc='center')
    table.set_fontsize(20)
    table.scale(1,3)
    ax.axis('off')
    
    plt.savefig("audioChart.png")
    plt.show()
