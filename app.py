from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")


app = Flask(__name__)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to remove repetitive sentences
def remove_repetitive_sentences(text):
    sentences = text.split('. ')
    seen_sentences = set()
    cleaned_story = []

    for sentence in sentences:
        if sentence not in seen_sentences:
            cleaned_story.append(sentence)
            seen_sentences.add(sentence)

    return '. '.join(cleaned_story).strip() + '.'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate-story', methods=['POST'])
def generate_story():
    # Get user input from the form
    character_name = request.form['character_name']
    place = request.form['place']
    theme = request.form['theme']

    # Formulate the prompt for GPT-2
    prompt = f"Once upon a time in {place}, there was a child named {character_name}. The story is about {theme}. Let's start the story!"
    
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
 
    # Generate attention mask
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    # Generate the story using GPT-2
    outputs = model.generate(inputs, 
                             max_length=70, 
                             num_return_sequences=1,                             
                             pad_token_id=tokenizer.eos_token_id,
                             temperature=0.7,  # Controls randomness. Lower values make output more focused.
                             top_p=0.9,  # Controls diversity via nucleus sampling.
                             do_sample=True,
                             repetition_penalty=1.2 ) # Penalize repeating the same word.)

    

    # Decode the generated tokens into text
    story = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

    # Post-process the story to remove repetitive phrases (optional step)
    story = remove_repetitive_sentences(story)

    # Render the result on the same page
    return render_template('index.html', story=story, character_name=character_name, place=place, theme=theme)

if __name__ == "__main__":
    app.run(debug=True)
