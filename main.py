import graphs
import io
import os
import json
import numpy
import cv2
import pyaudio
import wave
import matplotlib.pyplot as plt

from google.cloud import language_v1
from google.cloud.language_v1 import enums as fish
from google.cloud.speech_v1 import enums

from google.cloud import speech_v1
from google.cloud import vision
from google.cloud.vision import types

credential_path = "CREDENTIALFILENAME.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
clientVision = vision.ImageAnnotatorClient()
clientSpeech = speech_v1.SpeechClient()
clientLanguage = language_v1.LanguageServiceClient()

def recordAudio():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 16000  # Record at 44100 samples per second
    seconds = 3
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    audio = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, 20):
        data = stream.read(chunk)
        audio.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(audio))
    wf.close()

def convertToWave(audioFiles, p):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 16000  # Record at samples per second
    seconds = 3
    qIter = 0
    for audio in audioFiles:
        #print((audioFiles))
        fileAudioName  = os.path.dirname(__file__) + "/Audio/" + str(qIter+1) + "/output.wav"
        #print(fileAudioName)
        os.makedirs(os.path.dirname(fileAudioName), exist_ok=True)
        #wf = wave.open(fileAudioName, 'wb')
        with wave.open(fileAudioName, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(audio))
            wf.close()
        qIter = qIter + 1;

def __draw_label(img, text, pos):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    color = (255, 255, 255)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    #cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def recordInfo():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 16000  # Record at samples per second
    seconds = 3
    ###update in convert to wave if needed too###
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    questions = ['How do you feel today?', 'Are you feeling hungry?', 'Do you think you are understood in this world?']
    imageFolder = os.path.dirname(__file__) + "/Pictures/Output"
    cap = cv2.VideoCapture(0)

    i = 0
    frameId = 1
    qIter = 0

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    audio = []  # Initialize array to store frames
    audioFiles = []
    while(True):
        # Capture frame-by-frame
        data = stream.read(chunk)
        audio.append(data)
        #print(len(audio))

        ret, frame = cap.read()
        __draw_label(frame, questions[qIter], (20, 50))
        #print(str(i) + " " + str(frameId))
        if (i < 30):
            i = i + 1
        else:
            filename = imageFolder + str(qIter+1) + "/" + str(frameId) + ".jpg"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            #print(filename)
            cv2.imwrite(filename, frame)
            frameId = frameId + 1
            i = 0

        # Display the resulting frame
        cv2.imshow('Evaluation',frame)

        if cv2.waitKey(1) & 0xFF == ord('\r'):
            audioFiles.append(audio)
            #print((audio))
            audio = []

            if (qIter >= 2):
                stream.stop_stream()
                stream.close()
                p.terminate()
                break
            else:
                qIter = qIter + 1
                frameId = 1;



    cap.release()
    cv2.destroyAllWindows()

    #print(audioFiles)
    convertToWave(audioFiles, p)

def detectVideoEmotion(folder):
    data = dict()
    #print(folder)
    images = []
    totalSum = 0
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), folder)):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg']]):
            #print(filename)
            with io.open(os.path.join(os.path.dirname(__file__), folder,filename),'rb') as image_file:

                content = image_file.read()
                image = vision.types.Image(content=content)
                response = clientVision.face_detection(image=image)
                faces = response.face_annotations

                likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                                   'LIKELY', 'VERY_LIKELY')
                emotions = dict()
                for face in faces:
                    #print('sorrow: {}'.format(likelihood_name[face.sorrow_likelihood]))
                    emotions['sorrow'] = (likelihood_name[face.sorrow_likelihood])
                    emotions['angry'] = (likelihood_name[face.anger_likelihood])
                    #print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
                    emotions['joy'] = (likelihood_name[face.joy_likelihood])

                number = -0.2
                for emotion, likelihood in emotions.items():
                    if likelihood == "VERY_LIKELY":
                        if emotion == 'sorrow':
                            number = -0.8
                        elif emotion == 'joy':
                            number = +0.8
                    elif likelihood == "LIKELY":
                        if emotion == 'sorrow':
                            number = -0.6
                        elif emotion == 'joy':
                            number = +0.6
                    elif likelihood == "POSSIBLE":
                        if emotion == 'sorrow':
                            number = -0.6
                        elif emotion == 'joy':
                            number = +0.4
                    elif likelihood == "UNLIKELY":
                        if emotion == 'sorrow':
                            number = -0.4
                        elif emotion == 'joy':
                            number = +0.3
                goodPart, extension = os.path.splitext(filename)
                data[int(goodPart)] = (number)
                totalSum = totalSum + number
                print(emotions)
                print(number)
                #with open('appData.json', 'w') as fp:
                    #fp.write(json.dumps(data[filename]))
    sortedData = dict()
    for k in sorted(data.keys()):
        sortedData[k] = data[k]

    totalSum = totalSum / len(data.keys())
    print(str(totalSum) + "  " + folder)
    return sortedData

def detectSpeech(local_file_path):
    language_code = "en-US"
    sample_rate_hertz = 16000

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
             "language_code": language_code,
             "sample_rate_hertz": sample_rate_hertz,
             "encoding": encoding,
             "audio_channel_count" : 2
            }

    with io.open(local_file_path + "/output.wav", "rb") as f:
        content = f.read()
    audio = {"content": content}

    response = clientSpeech.recognize(config, audio)
    print("got a response")
    print(response)
    output = []
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        output.append(alternative.transcript)
        print(alternative)
        print(u"Transcript: {}".format(alternative.transcript))
    return output

def detectSpeechEmotion(text_content):
    type_ = fish.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type": type_, "language": language}
    encoding_type = fish.EncodingType.UTF8

    response = clientLanguage.analyze_sentiment(document, encoding_type=encoding_type)
    # Get overall sentiment of the input document
    print(u"Document sentiment score: {}".format(response.document_sentiment.score))
    print(
            u"Document sentiment magnitude: {}".format(
                response.document_sentiment.magnitude
                )
            )
    # Get sentiment for all sentences in the document
    for sentence in response.sentences:
        print(u"Sentence text: {}".format(sentence.text.content))
        print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        print(sentence.sentiment)
        print(u"Sentence sentiment magnitude: {}".format(sentence.sentiment.magnitude))

    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    print(u"Language of the text: {}".format(response.language))
    return response.document_sentiment.score

def main():
    #recordAudio()
    recordInfo()
    fullData = dict()
    for folder in os.listdir(os.path.join(os.path.dirname(__file__), 'Pictures')):
        fullData[folder] = (detectVideoEmotion(os.path.join('Pictures',folder)))
    with open('appFaceData.json', 'w') as fp:
        fp.write(json.dumps(fullData))
    print (fullData)
    scores = dict()
    for folder in os.listdir(os.path.join(os.path.dirname(__file__), 'Audio')):
        strList = (detectSpeech(os.path.join(os.path.dirname(__file__),'Audio',folder)))
        sentiments = []
        for item in strList:
            #print(item)
            sentiments.append(detectSpeechEmotion(item))
        sum = 0
        for item in sentiments:
            sum = sum + item
        sum = sum / len(sentiments)
        goodpart, badpart = os.path.splitext(folder)
        #print(goodpart)
        scores[(goodpart)] = sum
        #print(sum)
    sortedSpeechData = dict()
    for k in sorted(scores.keys()):
        sortedSpeechData[k] = scores[k]
    with open("appSpeechData.json", "w") as fp:
        fp.write(json.dumps(sortedSpeechData))
    #print(sortedSpeechData)

    graphs.makeGraph()

main()
