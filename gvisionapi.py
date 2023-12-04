import os, sys
sys.path.append('/root/workspace')

from google.cloud import vision
from chatbots.claude import Claude, models, ASK_TEMPLATE
from config import *

bot = Claude(model=models[0], temperature=0.1, system_prompt=ASK_TEMPLATE, stream=False)

def detect_labels(image_path):
    """Detects labels in the image file."""
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    result = "\n_________________{}__________________\n".format(image_path)

    for text in texts[1:]:
        # print('\n"{}"'.format(text.description))
        result += text.description

    return result

folder_path = '/root/workspace/ex_data/result'
file_list = os.listdir(folder_path)

result = ""

for file_name in file_list:
    result += detect_labels(os.path.join(folder_path, file_name))

processed_data = ""

for response in bot.ask_llm("{} \nRewrite it with good spacing and line breaks.".format(result), stream=True):
    processed_data += response

with open(folder_path+'/result.txt', 'w', encoding='utf-8') as writer:
    writer.write(processed_data)