import io
import os
import glob
import json

from google.cloud import vision
from google.cloud.vision import types

credential_path = "HACKRU-726f36ed559e.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
client = vision.ImageAnnotatorClient()


def detectEmotion(folder):
    data = dict()
    #print(folder)
    images = []
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), folder)):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg']]):
            #print(filename)
            with io.open(os.path.join(os.path.dirname(__file__), folder,filename),'rb') as image_file:

                content = image_file.read()
                image = vision.types.Image(content=content)
                response = client.face_detection(image=image)
                faces = response.face_annotations

                likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                                   'LIKELY', 'VERY_LIKELY')
                emotions = dict()
                for face in faces:
                    #print('sorrow: {}'.format(likelihood_name[face.sorrow_likelihood]))
                    emotions['sorrow'] = (likelihood_name[face.sorrow_likelihood])
                    #print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
                    emotions['anger'] = (likelihood_name[face.anger_likelihood])
                    #print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
                    emotions['joy'] = (likelihood_name[face.joy_likelihood])

                goodPart, extension = os.path.splitext(filename)
                data[int(goodPart)] = (emotions)
                #with open('appData.json', 'w') as fp:
                    #fp.write(json.dumps(data[filename]))
    sortedData = dict()
    for k in sorted(data.keys()):
        sortedData[k] = data[k]

    return sortedData

#check Pictures Folder
def main():
    fullData = dict()
    for folder in os.listdir(os.path.join(os.path.dirname(__file__), 'Pictures')):
        fullData[folder] = (detectEmotion(os.path.join('Pictures',folder)))

    #fullDataJSON = json.dumps(fullData)
    with open('appFaceData.json', 'w') as fp:
        fp.write(json.dumps(fullData))
    print (fullData)

main()