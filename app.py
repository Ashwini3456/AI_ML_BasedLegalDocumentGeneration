import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from Summarizer import Summariser

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,session,flash,redirect, url_for, session,flash
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, Dense
from tensorflow.keras.models import Sequential, load_model
import csv

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


app = Flask(__name__)
app.secret_key = '1a2b3c4d5e'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'system'
app.config['MYSQL_DB'] = 'accounts'

# Intialize MySQL
mysql = MySQL(app)


def generate2_summary2(input_text):
    # Load the tokenizer
    with open('idf.pkl', 'rb') as f:
        s_tokenizer = pickle.load(f)
    
    # Load the model
    enc_model = tf.keras.models.load_model('model.h5')
    dec_model = tf.keras.models.load_model('model.h5')
    
    # Tokenize the input text
    input_seq = s_tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=800, padding='post')
    
    # Generate the summary
    h, c = enc_model.predict(input_seq)
    
    next_token = np.zeros((1, 1))
    next_token[0, 0] = s_tokenizer.word_index['sostok']
    output_seq = ''
    
    stop = False
    count = 0
    
    while not stop:
        if count > 100:
            break
        decoder_out, state_h, state_c = dec_model.predict([next_token]+[h, c])
        token_idx = np.argmax(decoder_out[0, -1, :])
        
        if token_idx == s_tokenizer.word_index['eostok']:
            stop = True
        elif token_idx > 0 and token_idx != s_tokenizer.word_index['sostok']:
            token = s_tokenizer.index_word[token_idx]
            output_seq = output_seq + ' ' + token
        
        next_token = np.zeros((1, 1))
        next_token[0, 0] = token_idx
        h, c = state_h, state_c
        count += 1
        
    return output_seq.strip()

def replace_specific_words_in_square_brackets(text, replacement, words_to_replace):
    # Regular expression pattern to match specific words inside square brackets
    pattern = r'\[({})\]'.format('|'.join(map(re.escape, words_to_replace)))
    
    # Define a function for replacement
    def replace(match):
        return replacement
    
    # Use re.sub() to perform the replacement
    result = re.sub(pattern, replace, text)
    return result


def re_clean(text):
    import re
    text = re.sub(r'https?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', ' ', text) 
    text = re.sub(r'[_\-;%()|+&=*%:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text=re.sub(r'\n',' ',text)
    text=re.sub(' est ',' ',text)
    text=re.sub(r'[?!]','.',text)
    return text
#used to expand contractions
def expand(text):
    import contractions
    text=text.split()
    final=[]
    for word in text:
        try:
            final.append(contractions.fix(word)+" ") 
        except:
            final.append(word+" ")
            print(word)
    return "".join(final)
#used to remove small useless sentences
def remove(texts):
    final=[]
    for text in texts:
        sents=[]
        sentences=text.split(".")
        for sentence in sentences:
            if(len(sentence.split())>=5):
                sents.append(sentence+".")
        final.append("".join(sents))
    return final
#used to remove Common news tags
def removeTag(texts):
    final=[]
    #removing cnn and est
    for text in texts:
        cnn=text.find("cnn")
        if(cnn!=-1 and cnn<len(text)//10):
            text=text[cnn+3:]
        found=False
        for i in range(2):
            est=text.find(" est,")
            if(est<len(text)//5 and est!=-1):
                text=text[est+5:]
                found=True
        fs=text.find(".")
        if(fs<20 and fs!=-1 and found):
            text=text[fs:]
        final.append(text)
    return final
# Calculate tf idf scores for sentences
def tfidf(sentences,d):
    scores=[0]*len(sentences)
    freq={}
    total=0
    for sentence in sentences:
        words=sentence.split()
        for i in words:
            total+=1
            if(i in freq):
                freq[i]+=1
            else:
                freq[i]=1
    for i in freq:
        freq[i]/=total
    wordlengths=[]
    for i in range(len(sentences)):
        words=sentences[i].split()
        n=len(words)
        wordlengths.append(n)
        score=0
        for word in words:
            if(word in d):
                score+=(freq[word]*d[word])
            else:
                score+=(freq[word]*1)
        score= score/n if n>0 else 0
        scores[i]=score
    
    l=[[scores[i],sentences[i],wordlengths[i]] for i in range(len(sentences))]
    l.sort(reverse=True)
    return l
# extract sentences
def extract(text,no_words):
    import pickle
    text=expand(re_clean(text)).lower()
    cleaned=removeTag(remove([text]))[0]
    d=0
    with open("idf.pkl", 'rb') as fp:
        d = pickle.load(fp)
    text=text.split(".")
    scores=tfidf(text,d)
    final=[]
    no_words-=scores[0][2]
    while(no_words>0 and len(scores)>1):
        sentence=scores.pop()[1]+"."
        place=-1
        for i in range(len(sentence)):
            if(sentence[i].isalpha()):
                place=i
                break
        sentence=sentence[:place]+sentence[place].upper()+sentence[place+1:] if place>-1 else sentence
        final.append("".join(sentence))
        no_words-=scores[0][2]
    
    final="".join(final)
    import re
    final=re.sub(r"you.","U.",final)
    return final

model = load_model("model.h5")
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
max_words = 10000
max_len = 100
embedding_dim = 100
summary_max_len = 19  # Adjust according to your summary length
tokenizer = Tokenizer(num_words=max_words)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [porter.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

# Example of prediction
def predict_summary(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(sequence)

    # Decode the predicted sequence to get the summary
    decoded_summary = [tokenizer.index_word.get(num, '') for num in np.argmax(prediction[0], axis=-1) if num != 0]
    return ' '.join(decoded_summary)


@app.route('/', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            #session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return render_template('main.html',title="Food Time ")#redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            flash("Incorrect username/password!", "danger")
    return render_template('login.html',title="Login")



@app.route('/register', methods=['GET', 'POST'])
def register():
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM accounts WHERE username = %s', (username))
        cursor.execute( "SELECT * FROM accounts WHERE username LIKE %s", [username] )
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        else:
        # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (%s, %s, %s)', (username,email, password))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
            return render_template('login.html',title="Login")

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    return render_template('login.html',title="Register")

def capitalize_each_word(sentence):
    # Split the sentence into words
    words = sentence.split()
    
    # Capitalize each word and join them back into a sentence
    capitalized_sentence = ' '.join(word.capitalize() for word in words)
    
    return capitalized_sentence

@app.route('/home', methods=['GET'])
def index():
    # Main page
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("test of food")
    if request.method == 'POST':
        # Get the file from post request
        fpn = request.form['fpn']
        fan=request.form['fan']
        fpa=request.form['fpa']
        spn=request.form['spn']
        san=request.form['san']
        spa=request.form['spa']
        ta=request.form['ta']

        if(ta=='0'):
            print("rent")
            file_path = 'rents.csv'
            csv_data = read_csv_file(file_path)
            s="This rental agreement is entered into between "+fpn +", residing at "+ fpa+ " , hereinafter referred to as the Landlord", "and "+spn +" , residing at "+spa+" , hereinafter referred to as the Tenant"
            #print(str(csv_data[1]))
            paragraph = Summariser.clean_text(str(csv_data[1]))
            #print(len(str(csv_data[2])))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s[0])
            s1 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)+"with Section 108(l) of the Transfer of Property Act, 1882"
            #print(summary)
            print(len(s1))
            paragraph = Summariser.clean_text(str(csv_data[3][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s1)
            s2 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)+"with Indian Easements Act, 1882"
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[4][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s2)
            s3 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)+" Section 105 of the Transfer of Property Act, 1882"
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[5][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)

            predict_summary(s3)
            s4 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)+" with Section 106 of the Transfer of Property Act, 1882"
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[6][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)

            predict_summary(s4)
            s5 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)+" with Section 108(a) of the Transfer of Property Act, 1882"
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[7][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s5)
            s6 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)+" Section 108(e) of the Transfer of Property Act, 1882"
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[8][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s6)
            s7 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)+" Indian Easements Act, 1882"
            #print(summary)
            predict_summary(s7)
            from datetime import date
            tdate = date.today()
            import random
            docid='DOC/MCA/'+str(random.randint(20, 60))

	
            return render_template('result.html',ta="Rent Agreement",s=s,s1=capitalize_each_word(s1),s2=capitalize_each_word(s2),s3=capitalize_each_word(s3),s4=capitalize_each_word(s4),s5=capitalize_each_word(s5),s6=capitalize_each_word(s6),s7=capitalize_each_word(s7),fpn=fpn, spn=spn,fan=fan,san=san,tdate=tdate,docid=docid)
        if(ta=='1'):
            file_path = 'partnership.csv'
            csv_data = read_csv_file(file_path)
            s="This Partnership agreement is entered into between "+fpn +", residing at "+ fpa+ " , hereinafter referred to as the Partner ", "and "+spn +" , residing at "+spa+" , hereinafter referred to as the Partner"
            paragraph = Summariser.clean_text(str(csv_data[2]))
            #ptint(len(str(csv_data[2])))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s[0])
            s1 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            print(len(s1))
            paragraph = Summariser.clean_text(str(csv_data[3][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s1)
            s2 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[4][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s2)
            s3 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[5][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s3)
            s4 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[6][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s4)
            s5 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[7][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s5)
            s6 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[8][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s6)
            s7 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            predict_summary(s7)
            from datetime import date
            tdate = date.today()
            import random
            docid='DOC/MCA/'+str(random.randint(20, 60))

	
            return render_template('result.html',ta="Partnership Agreement",s=s,s1=capitalize_each_word(s1),s2=capitalize_each_word(s2),s3=capitalize_each_word(s3),s4=capitalize_each_word(s4),s5=capitalize_each_word(s5),s6=capitalize_each_word(s6),s7=capitalize_each_word(s7),fpn=fpn, spn=spn,fan=fan,san=san,tdate=tdate,docid=docid)

        if(ta=='2'):
            file_path = 'loan.csv'
            csv_data = read_csv_file(file_path)
            s="This Loan agreement is entered into between "+fpn +", residing at "+ fpa+ " , hereinafter referred to as the Barrower", "and "+spn +" , residing at "+spa+" , hereinafter referred to as the Lender"
            paragraph = Summariser.clean_text(str(csv_data[2]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s[0])
            s1 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[3][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s1)
            s2 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[4][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s2)
            s3 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[5][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s3)
            s4 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[6][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s4)
            s5 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[7][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s5)
            s6 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            paragraph = Summariser.clean_text(str(csv_data[8][0]))
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            #print(sentence_scores)
            threshold = Summariser.find_average_score(sentence_scores)
            #print(threshold)
            predict_summary(s6)
            s7 = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            #print(summary)
            predict_summary(s7)
            from datetime import date
            tdate = date.today()
            import random
            docid='DOC/MCA/'+str(random.randint(20, 60))

	
            return render_template('result.html',ta="Loan Agreement",s=s,s1=capitalize_each_word(s1),s2=capitalize_each_word(s2),s3=capitalize_each_word(s3),s4=capitalize_each_word(s4),s5=capitalize_each_word(s5),s6=capitalize_each_word(s6),s7=capitalize_each_word(s7),fpn=fpn, spn=spn,fan=fan,san=san,tdate=tdate,docid=docid)
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=False,host='0.0.0.0')
