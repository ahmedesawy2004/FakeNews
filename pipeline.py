import os
import joblib
import numpy as np
from PIL import Image

# Import ML frameworks (uncomment/adjust based on what you used)
# import torch
# import torchvision.transforms as transforms
# import tensorflow as tf

class MultimodalPipeline:
    def __init__(self):
        # Paths to your saved models
        self.nlp_model_path = "nlp_model.pkl"
        self.vision_model_path = "vision_model.pth" # or .keras / .h5
        
        # Load models
        self.nlp_model = self._load_nlp_model()
        self.vision_model = self._load_vision_model()
        
        # Define expected label mapping: 0 -> Real, 1 -> Fake
        self.classes = {0: "Real", 1: "Fake"}

    def _load_nlp_model(self):
        """Loads the NLP model (Scikit-learn LogisticRegression and TF-IDF Vectorizers)."""
        if os.path.exists("models/char_tfidf.joblib") and os.path.exists("models/word_tfidf.joblib") and os.path.exists("models/logisticregression_model.joblib"):
            try:
                self.char_tfidf = joblib.load("models/char_tfidf.joblib")
                self.word_tfidf = joblib.load("models/word_tfidf.joblib")
                self.clf = joblib.load("models/logisticregression_model.joblib")
                print("Loaded NLP models from 'models' folder")
                return True
            except Exception as e:
                print(f"Error loading NLP model: {e}")
        else:
            print("Warning: NLP models not found in 'models' directory. Using dummy model.")
        self.char_tfidf = None
        self.word_tfidf = None
        self.clf = None
        return False

    def _load_vision_model(self):
        """Loads the Computer Vision model (e.g., PyTorch or Keras model)."""
        if os.path.exists(self.vision_model_path):
            try:
                # --- Example for PyTorch ---
                # import torchvision.models as models
                # model = models.resnet18(pretrained=False)
                # model.fc = torch.nn.Linear(model.fc.in_features, 2)
                # model.load_state_dict(torch.load(self.vision_model_path))
                # model.eval()
                # return model
                
                # --- Example for Keras/TensorFlow ---
                # return tf.keras.models.load_model(self.vision_model_path)
                
                print(f"Vision model loading code is commented out. Check `pipeline.py`.")
            except Exception as e:
                print(f"Error loading Vision model: {e}")
        else:
            print(f"Warning: Vision model not found at {self.vision_model_path}. Using dummy model.")
        return None

    def predict_text(self, text):
        """Returns the probability that the text is Fake (0.0 to 1.0)."""
        if not text or not text.strip():
            return None
            
        if self.clf and self.word_tfidf and self.char_tfidf:
            try:
                from scipy.sparse import hstack
                word_features = self.word_tfidf.transform([text])
                char_features = self.char_tfidf.transform([text])
                X = hstack([word_features, char_features])
                proba = self.clf.predict_proba(X)[0]
                
                raw_fake_prob = float(proba[0])
                
                # Calibrate the probability to handle the model's skewed intercept (-1.77) for short user inputs
                # Empty/short text defaults to ~0.8546 raw probability. We center this at 0.5 (neutral)
                baseline = 0.8546
                if raw_fake_prob >= baseline:
                    calibrated_prob = 0.5 + 0.5 * ((raw_fake_prob - baseline) / (1.0 - baseline))
                else:
                    calibrated_prob = 0.5 * (raw_fake_prob / baseline)
                    
                return calibrated_prob
            except Exception as e:
                print(f"Error during text prediction: {e}")
        
        # Dummy fallback: simplistic heuristic for demonstration
        fake_keywords = ["shocking", "you won't believe", "secret", "hoax", "revealed"]
        score = sum(0.2 for word in fake_keywords if word in text.lower())
        return min(score + 0.1, 0.95)

    def predict_image(self, image):
        """Returns the probability that the image is Fake (0.0 to 1.0)."""
        if image is None:
            return None
            
        if self.vision_model:
            try:
                # --- PyTorch Example ---
                # transform = transforms.Compose([
                #     transforms.Resize((224, 224)),
                #     transforms.ToTensor(),
                #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # ])
                # img_tensor = transform(image).unsqueeze(0)
                # with torch.no_grad():
                #     outputs = self.vision_model(img_tensor)
                #     probabilities = torch.nn.functional.softmax(outputs, dim=1)
                #     return float(probabilities[0][1]) # Assuming index 1 is Fake
                
                # --- Keras/TensorFlow Example ---
                # img = image.resize((224, 224))
                # img_array = np.array(img) / 255.0
                # img_array = np.expand_dims(img_array, axis=0)
                # preds = self.vision_model.predict(img_array)
                # return float(preds[0][1]) # Or preds[0][0] if binary crossentropy output
                pass
            except Exception as e:
                print(f"Error during image prediction: {e}")

        # Dummy fallback: random score based on image size for demonstration
        np.random.seed(image.size[0] + image.size[1])
        return float(np.random.uniform(0.1, 0.9))

    def predict(self, text, image):
        """
        Computes multimodal prediction.
        Returns:
            dict with individual scores, combined score, and final verdict.
        """
        text_fake_prob = self.predict_text(text)
        image_fake_prob = self.predict_image(image)
        
        # Handle cases where one or both modalities are missing
        if text_fake_prob is None and image_fake_prob is None:
            return None
            
        if text_fake_prob is None:
            combined_score = image_fake_prob
            logic = "Image Only"
        elif image_fake_prob is None:
            combined_score = text_fake_prob
            logic = "Text Only"
        else:
            # Multimodal combination logic
            # E.g., Weighted average (say text is 40%, image is 60%)
            # Or flag as high risk if either modality shows high fake probability
            
            # Simple average
            # combined_score = (text_fake_prob + image_fake_prob) / 2
            
            # Heuristic Logic: Highest confidence wins (MAX pooling)
            # This means if an image is definitely fake but text looks normal, we flag it.
            combined_score = max(text_fake_prob, image_fake_prob)
            logic = "Multimodal (Max Risk Logic)"
        
        # Determine verdict based on threshold
        threshold = 0.5
        if combined_score >= 0.75:
            verdict = "High Risk / Fake News"
        elif combined_score >= 0.5:
            verdict = "Suspicious / Misleading"
        else:
            verdict = "Likely Real News"

        return {
            "text_score": text_fake_prob,
            "image_score": image_fake_prob,
            "combined_score": combined_score,
            "verdict": verdict,
            "logic": logic
        }
