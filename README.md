# Emotion Detection


## Description
Python 3.6 application that records video via OpenCV and audio with Pyaudio and analyzes emotion via Google Cloud API for face emotion detection, speech to text, and text emotion detection. 

Purpose is to determine emotional well-being of psychiatric patients and their response to various questions overlayed on an OpenCV video capture with pyaudio capturing audio in the background via computer microphone.

Final display includes plot of emotional status over time related to the questions asked along with specific facial and audio data

## Dependencies
Requires Google Cloud Authorization and credential JSON within file folder and a variety of python libraries:
- OpenCV and dependencies
- Pyaudio
- Matplotlib
- Numpy
- Wave
- Google-cloud and its dependencies:
	- Google-cloud-vision
  - Google-cloud-speech
  - Google-cloud-language
