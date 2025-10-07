from flask import Flask, redirect, render_template, request, url_for
from model import predict_from_url
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    if not url:
        return "No URL provided", 400
    try:
        result = predict_from_url(url)
        return render_template('index.html', prediction=result, url=url)
    except Exception as e:
        return str(e), 500
    
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    # Check if a file was posted
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        # Use secure_filename to prevent security issues
        filename = secure_filename(file.filename)
        
        # It's good practice to save the file temporarily, but for this use case,
        # we can read it directly from the file stream.
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file)

            # Check if the required 'url' column exists
            if 'url' not in df.columns:
                # Handle error - column not found
                # You can pass an error message to the template
                return render_template('index.html', error="CSV must have a column named 'url'")

            # Process each URL and get predictions
            results = []
            for index, row in df.iterrows():
                url = row['url']
                prediction = predict_from_url(url)
                results.append({'url': url, 'prediction': prediction})
            
            # Render the page with the table of results
            return render_template('index.html', csv_results=results)

        except Exception as e:
            # Handle potential errors with file reading or processing
            return render_template('index.html', error=str(e))

    # If file is not a CSV, redirect or show an error
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(port=3000, debug=True)

