from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

# Load the BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)



@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method =='POST':
       input_text = request.form['text']

       input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)
       summary_ids = model.generate(input_ids, max_length=120, min_length=80, length_penalty=2.0, num_beams=4, early_stopping=True)

       summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

       return render_template('summarizer.html', input_text=input_text, summary=summary)

    return render_template('summarizer.html')



if __name__=='__main__':
   #app.debug = True
   app.run(debug = True)