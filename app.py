from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import json

app = Flask(__name__)

# Load the pre-trained model and tokenizer for summarization
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def get_word_count(text):
    words = text.split()
    return len(words)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/summarize", methods=['POST'])
def summarize():
    try:
        # Get the input text from the user
        text = request.form['text']
        # Get the desired summary length from the user
        summary_length = int(request.form['summary_length'])

        # Check if the desired summary length is at least 30 words
        if summary_length < 30:
            return jsonify({'error': 'Desired summary length should be at least 30 words.'})

        # Generate the summary using the T5 model
        generated_summary = summarizer(text, max_length=summary_length, min_length=30, do_sample=False)

        # Calculate word counts
        input_word_count = get_word_count(text)
        summary_word_count = get_word_count(generated_summary[0]['summary_text'])

        # Store data in a dictionary
        data = {
            'input_text': text,
            'input_word_count': input_word_count,
            'generated_summary': generated_summary[0]['summary_text'],
            'summary_word_count': summary_word_count
        }

        # Write data to a JSON file
        with open('data.json', 'a') as file:
            json.dump(data, file)
            file.write('\n')

        # Return the summary and word counts as a JSON response
        return jsonify({
            'summary': generated_summary[0]['summary_text'],
            'input_word_count': input_word_count,
            'summary_word_count': summary_word_count,
            'input_text': text,
            'generated_summary': generated_summary[0]['summary_text']
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)