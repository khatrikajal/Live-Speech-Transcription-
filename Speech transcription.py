import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import time

# create an instance of the recognizer
r = sr.Recognizer()

# define the microphone as source
mic = sr.Microphone()

# adjust the energy threshold
r.energy_threshold = 500

# download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# load the English language model for spaCy
nlp = spacy.load('en_core_web_sm')

# define a function for text summarization
def summarize_text(Text):
    # split the text into sentences
    sentences = sent_tokenize(Text)

    # remove stop words and punctuations from sentences
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
        filtered_sentences.append(' '.join(filtered_words))

    # combine filtered sentences into a summary
    summary = ' '.join(filtered_sentences)

    # print the summary
    print("Summary:")
    print(summary)
    return summary

# define a function for speaker identification
def identify_speakers(transcription):
    # split the transcription into sentences
    sentences = sent_tokenize(transcription)

    # create a dictionary to map speakers to their sentences
    speaker_sentences = {}

    # loop through each sentence
    for sentence in sentences:
        # create a Doc object with spaCy
        doc = nlp(sentence)

        # check if the sentence contains a speaker tag
        for token in doc:
            if token.pos_ == "NOUN" and token.text.lower() in ["speaker", "voice"]:
                # get the speaker label
                speaker_label = doc[0].text

                # add the sentence to the speaker's list of sentences
                if speaker_label in speaker_sentences:
                    speaker_sentences[speaker_label].append(sentence)
                else:
                    speaker_sentences[speaker_label] = [sentence]
                break  # assume only one speaker tag per sentence

    # if there are no speaker labels, assume single speaker
    if not speaker_sentences:
        speaker_sentences["Speaker"] = sentences

    # create a list of speaker labels and their sentences
    speaker_labels = []
    for i, (speaker_label, sentences) in enumerate(speaker_sentences.items()):
        for sentence in sentences:
            speaker_labels.append(f"Speaker {i+1}: {sentence}")

    # print the speaker labels
    print("Speaker labels:")
    print(speaker_labels)
    return list(speaker_sentences.keys())

# Modify the extract_keywords function to return the top 5 keywords
def extract_keywords(Text):
    # Create a Doc object with spaCy
    doc = nlp(Text)

    # Create a list of candidate keywords
    candidate_keywords = []
    for chunk in doc.noun_chunks:
        if len(chunk) > 1 and not any(token.pos_ == "PRON" for token in chunk):
            candidate_keywords.append(chunk.text)

    # Create a dictionary of keyword frequencies
    keyword_frequencies = {}
    for keyword in candidate_keywords:
        keyword_frequencies[keyword] = Text.count(keyword)

    # Sort the keywords by frequency
    sorted_keywords = sorted(keyword_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Extract the top 5 keywords
    top_keywords = [keyword for keyword, _ in sorted_keywords[:5]]

    # Print the top 5 keywords
    print("Top 5 keywords:")
    for keyword in top_keywords:
        print(keyword)

    return top_keywords


# define a function for writing to a text file
def write_to_file(filename, data, keywords=None):
    with open(filename, 'a') as f:
        if data.startswith('Keywords') and keywords:
            # Extract the top five keywords and write them to the file
            top_keywords = keywords[:5]
            f.write('Keywords: ' + ', '.join(top_keywords) + '\n')
        else:
            f.write(data + '\n')
    if data.startswith('Transcription'):
        # extract speakers and write to file
        speakers = identify_speakers(data.split(':')[1].strip())
        write_to_file(filename, 'Speakers: ' + ', '.join(speakers))

# start the microphone
with mic as source:
    r.adjust_for_ambient_noise(source)
    print("Start speaking...")
    while True:
        # listen to the microphone and transcribe the speech
        audio = r.listen(source)
        Keywords = []
        text = ""  # define text variable outside try block
        try:
            text = r.recognize_google(audio, language='en-US', show_all=False)
            print(text)
            Summary = summarize_text(text)
            Speakers = identify_speakers(text)
            
# Inside your main loop
            Keywords = extract_keywords(text)
            write_to_file('output.txt', 'Transcription: ' + text)
            write_to_file('output.txt', 'Top 5 Keywords: ' + ', '.join(Keywords))

            
            write_to_file('output.txt', 'Summary: ' + Summary)
            write_to_file('output.txt', 'Speakers: ' + ', '.join(Speakers))
        except sr.UnknownValueError:
            print("Unable to recognize speech")
        except sr.RequestError as e:
            print("Request error: {0}".format(e))

        # Add a delay before listening again
        time.sleep(2)  # Adjust the delay duration as needed
