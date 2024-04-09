
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Initialize Firebase with your credentials
cred = credentials.Certificate(r'C:\Users\bhara\Downloads\Facial_Detection_Project-main\Facial_Detection_Project-main\serviceAccountKey.json')
firebase_admin.initialize_app(cred)

app = Flask(__name__)

# -----------------------------------------------------------------------------------------------

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analysis(input_text):
    encoded_text = tokenizer(input_text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    scores = [float(score) for score in scores]
    scores_dict = {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }
    return scores_dict

@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion():
    try:
        data = request.get_json()

        if 'text_data' not in data or 'user_name' not in data:
            return jsonify({'error': 'Missing required fields in the request payload'}), 400

        text_data = data['text_data']
        user_name = data['user_name']

        scores_dict = analysis(text_data)

        # Reference to the "users" collection
        users_ref = firestore.client().collection("users")
        # Reference to the specific user document
        user_doc_ref = users_ref.document(user_name)
        # Reference to the "emotions" subcollection within the user document
        emotions_ref = user_doc_ref.collection("textemotion")
        # Emotion data to be stored
        emotion_data = {"timestamp": firestore.SERVER_TIMESTAMP, "emotion": scores_dict}
        # Add emotion data to the "emotions" subcollection
        emotions_ref.add(emotion_data)
        print(scores_dict)

        return jsonify(scores_dict)

    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
