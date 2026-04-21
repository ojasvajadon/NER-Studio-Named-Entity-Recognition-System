# NER-Studio-Named-Entity-Recognition-System
NER Studio is an interactive Named Entity Recognition (NER) web application built using Flask and spaCy. It allows users to extract, visualize, and correct entities from unstructured text while continuously improving model accuracy through feedback-based learning.
features
Extract entities from text or uploaded .txt files
 Visualize entities using spaCy’s displacy
Hybrid approach:
Pre-trained + custom NER model
Rule-based accuracy improvements
Backup model support
Editable entity labels (human-in-the-loop)
Save corrections for retraining
Entity statistics & breakdown dashboard
Continuous learning pipeline               

system architure
User Input (Text/File)
        ↓
Flask Backend
        ↓
spaCy NER Model
        ↓
Rule-Based Enhancement
        ↓
Backup Model (Optional)
        ↓
Entity Visualization (displacy)
        ↓
User Feedback (Edit Labels)
        ↓
Save to JSON Dataset
        ↓
Model Retraining
        ↓
Improved Accuracy
