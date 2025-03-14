import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
file_path = "Student-Employability-Datasets.xlsx"
df = pd.read_excel(file_path)

# Define feature columns (modify if needed)
feature_cols = ['GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION',
                'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS',
                'COMMUNICATION SKILLS', 'Student Performance Rating']
target = 'CLASS'  # Change based on actual employability label

# Encode target variable
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# Prepare data for training
X = df[feature_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train MLP Classifier model
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# Gradio function to predict employability
def assess_employability(name, appearance, speaking, condition, alertness, confidence, ideas, communication, performance):
    input_data = np.array([[appearance, speaking, condition, alertness, confidence, ideas, communication, performance]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    
    if prediction == 1:
        return f"🚀 {name},💡 You're on the right path! Keep learning and growing—success is within reach! ✨"
    elif prediction == 0:
        return f"✨ {name},🌟 You're a rising star in the job market! Keep striving for greatness! 🚀"
    else:
        return f"📚 {name},📖 Growth takes time! Keep honing your skills, and you'll get there! 💪🚀"

# Gradio UI
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown("## 🌟 **Employability Predictor App** 🌟")
    gr.Markdown("### Rate your skills on a scale of 1 to 10")
    
    name = gr.Textbox(label="📝 Your Name")
    appearance = gr.Slider(1, 10, label="👔 General Appearance")
    speaking = gr.Slider(1, 10, label="🗣️ Manner of Speaking")
    condition = gr.Slider(1, 10, label="💪 Physical Condition")
    alertness = gr.Slider(1, 10, label="🧠 Mental Alertness")
    confidence = gr.Slider(1, 10, label="🔥 Self-Confidence")
    ideas = gr.Slider(1, 10, label="💡 Ability to Present Ideas")
    communication = gr.Slider(1, 10, label="🗨️ Communication Skills")
    performance = gr.Slider(1, 10, label="📈 Student Performance Rating")
    
    submit_btn = gr.Button("🚀 Check Employability 🚀")
    output = gr.Textbox(label="Result", interactive=True)
    
    submit_btn.click(assess_employability, inputs=[name, appearance, speaking, condition, alertness, confidence, ideas, communication, performance], outputs=output)

# Launch the app
if __name__ == "__main__":
    demo.launch(share = True)
