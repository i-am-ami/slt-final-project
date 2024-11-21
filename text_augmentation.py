from openai import OpenAI
import pandas as pd
import glob
import json
import concurrent.futures

# Replace with your OpenAI API key
client = OpenAI(
    api_key = 'REPLACE_WITH_YOUR_API_KEY'
)

# Function to extract data from JSON file
def extractFromJson(fName):
    with open(fName, 'r') as f:
        data = json.load(f)
    return data

# Function to post-process the model output
def postProcessText(x):
    x = [i.strip() for i in x]
    # get rid of all the punctuations
    punctuationList = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "<", ">", "'", '"']
    for i in range(len(x)):
        for punc in punctuationList:
            x[i] = x[i].replace(punc, "")
    # convert all words to uppercase
    for i in range(len(x)):
        x[i] = x[i].upper()
    return x

# Function to call the OpenAI model for augmentation
def getAugmentation(prompt, sentence):
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "I have this sentence: '" + sentence + "' " + prompt}],
        stream=True
    )
    res = []
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            res.append(chunk.choices[0].delta.content)
    x = "".join(res).split("|")
    x = postProcessText(x)
    return x

# Function to augment the text field of each JSON entry
def textAug(jsonData):
    prompt = "Your task is to edit the sentence. You can make the following edit to the sentence: \
        insertion, deletion, substitution, and multi-span editing, with the length of the edited text ranging from 1 word to 16 words. \
        Generate the edited sentence only. I want you to repeat the action 3 times, generating 3 new but different sentences on top of the original sentence.\
        Seperate each sentence with a '|' character."
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(getAugmentation, prompt, entry.get('text', '')) for entry in jsonData if entry.get('text', '')]
        for future, entry in zip(futures, jsonData):
            if entry.get('text', ''):
                augmented_texts = future.result()
                if 'augmented_texts' not in entry:
                    entry['augmented_texts'] = []
                entry['augmented_texts'].extend(augmented_texts)

# Function to process each file
def processFile(fName):
    print(f"Processing {fName}")
    data = extractFromJson(fName)
    textAug(data)
    # Overwrite the file with augmented data
    with open(fName, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Augmented data written to {fName}")

# Glob all the JSON files in the Json/ directory
fList = glob.glob('Json/*.json')

# Use ThreadPoolExecutor to parallelize the processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(processFile, fList)

# Example usage
if __name__ == "__main__":
    print("Text augmentation completed.")
