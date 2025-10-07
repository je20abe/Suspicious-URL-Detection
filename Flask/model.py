from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import joblib
import re
import os
from urllib.parse import urlparse
from tld import get_tld

# --- 1. Flask App Initialisation ---
app = Flask(__name__)

# --- 2. Load the Bundle ---
BUNDLE_PATH = "lgbm_bundle.joblib"  # Path to your model bundle

try:
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"] 
    feature_names = bundle.get("feature_names", [])
    versions = bundle.get("versions", {})
    
    print("Bundle loaded successfully.")
    print(f"Model sklearn version: {versions.get('sklearn', 'Unknown')}")
    print(f"Available classes: {label_encoder.classes_}")
    
except FileNotFoundError:
    model = None
    label_encoder = None
    print(f"CRITICAL: Bundle file '{BUNDLE_PATH}' not found.")
except Exception as e:
    model = None
    label_encoder = None
    print(f"Error loading bundle: {e}")

# --- 3. The All-in-One Feature Extractor ---
def extract_features_from_url(url: str) -> list:
    """
    Extracts the 21 features from a URL to match the trained model.
    The order of features in the returned list is critical.
    """
    # Ensure URL is a string
    if not isinstance(url, str):
        url = ""
        
    features = []
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc if parsed_url.netloc else ''
        path = parsed_url.path if parsed_url.path else ''

        # Feature 1: use_of_ip
        ip_match = re.search('(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
        features.append(1 if ip_match else 0)
        
        # Feature 2: abnormal_url
        features.append(1 if re.search(str(hostname), url) else 0)
        
        # Feature 3: count.
        features.append(hostname.count('.'))
        
        # Feature 4: count-www
        features.append(url.count('www'))
        
        # Feature 5: count@
        features.append(url.count('@'))
        
        # Feature 6: count_dir
        features.append(path.count('/'))
        
        # Feature 7: count_embed_domain
        features.append(path.count('//'))
        
        # Feature 8: short_url
        shortening_match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|po\.st|bc\.vc|twittthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|tlurl\.com|tweezer\.me|v\.gd|tr\.im|link\.zip\.net', url)
        features.append(1 if shortening_match else 0)
        
        # Feature 9: count-https
        features.append(url.count('https'))
        
        # Feature 10: count-http
        features.append(url.count('http'))
        
        # Feature 11: count%
        features.append(url.count('%'))
        
        # Feature 12: count?
        features.append(url.count('?'))
        
        # Feature 13: count-
        features.append(url.count('-'))
        
        # Feature 14: count=
        features.append(url.count('='))
        
        # Feature 15: url_length
        features.append(len(url))
        
        # Feature 16: hostname_length
        features.append(len(hostname))
        
        # Feature 17: sus_url
        sus_match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
        features.append(1 if sus_match else 0)
        
        # Feature 18: fd_length (first directory length)
        try:
            features.append(len(path.split('/')[1]))
        except:
            features.append(0)
        
        # Feature 19: tld_length
        try:
            tld = get_tld(url, fail_silently=True)
            features.append(len(tld) if tld else -1)
        except:
            features.append(-1)
        
        # Feature 20: count-digits
        features.append(sum(c.isdigit() for c in url))
        
        # Feature 21: count-letters
        features.append(sum(c.isalpha() for c in url))
        
    except Exception as e:
        print(f"Error extracting features for URL '{url}': {e}")
        # Return a list of zeros if feature extraction fails
        return [0] * 21
    
    return features

# --- 4. The Prediction Function ---
def predict_from_url(url: str):
    """Takes a URL, extracts features, and returns prediction with confidence."""
    if not model:
        return {"error": "Model not loaded"}
    
    if not label_encoder:
        return {"error": "Label encoder not loaded"}
    
    try:
        # Extract features from the URL
        extracted_features = extract_features_from_url(url)
        
        # Make prediction (returns numerical class)
        prediction = model.predict([extracted_features])
        numerical_result = prediction[0]
        
        # Get probability scores for confidence
        probabilities = model.predict_proba([extracted_features])[0]
        confidence = float(max(probabilities))
        
        # Convert numerical prediction back to class name using label encoder
        class_name = label_encoder.inverse_transform([numerical_result])[0]
        
        return {
            "prediction": class_name.capitalize(),
            "confidence": confidence,
            "url": url
        }
        
    except Exception as e:
        print(f"Error during prediction for URL '{url}': {e}")
        return {"error": "Prediction failed"}

# --- 5. Test function to verify everything is working ---
def test_prediction():
    """Test function to verify model and encoder are working"""
    if model and label_encoder:
        test_url = "http://google.com"
        result = predict_from_url(test_url)
        if "error" in result:
            print(f"Test failed: {result['error']}")
            return False
        else:
            print(f"Test prediction for '{test_url}': {result['prediction']} (confidence: {result['confidence']:.2f})")
            return True
    else:
        print("Cannot test - model or label encoder not loaded")
        return False

# Run test when app starts
if __name__ == '__main__':
    print("\n" + "="*50)
    print("FLASK APP INITIALISATION")
    print("="*50)
    test_prediction()
    print("="*50)
    app.run(debug=True)