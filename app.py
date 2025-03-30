from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
print("Loading model...")
pipe = pickle.load(open('pipe.pkl', 'rb'))
print("Model loaded successfully!")

# Define teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

@app.route('/')
def home():
    return render_template('index.html', teams=teams, cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from request
        data = request.get_json()
        print("Received Data:", data)  # Debugging

        # Extract form values
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        city = data['city']
        target = int(data['target'])
        score = int(data['score'])
        overs = float(data['overs'])
        wickets = int(data['wickets'])

        # Calculate additional parameters
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        remaining_wickets = 10 - wickets  # FIX: Renaming to match model's column
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        # Create input DataFrame
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'remaining_wickets': [remaining_wickets],  # ✅ FIXED COLUMN NAME
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        print("Input DataFrame:", input_df)  # Debugging

        # Predict probabilities
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Return JSON response
        return jsonify({
            'win_probability': round(win * 100, 2),
            'lose_probability': round(loss * 100, 2)
        })

    except Exception as e:
        print("Error in Prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
