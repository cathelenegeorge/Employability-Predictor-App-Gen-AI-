# Employability-Predictor-App-(Gen-AI)
This project uses a Multi-Layer Perceptron (MLP) to predict student employability based on skills and performance. A Gradio-powered web app provides real-time AI-driven assessments with motivational feedback. 🚀

Here's the updated project description with **MLP (Multi-Layer Perceptron)** as the model instead of a simple Perceptron:  

---

# **Employability Prediction using Multi-Layer Perceptron (MLP)**

## **Project Overview**  
This project leverages a **Multi-Layer Perceptron (MLP) neural network** to predict a student's employability based on their skills and performance. The model is trained on a dataset containing students' ratings in various **soft skills, communication abilities, and academic performance**. The primary goal is to provide an **AI-driven assessment** of employability and offer motivational feedback to users.  

An **interactive Gradio web app** is integrated to make the prediction process user-friendly, allowing individuals to input their ratings and receive instant insights.

## **Key Features**
- **Advanced Machine Learning Model:**  
  Uses **MLP (Multi-Layer Perceptron)** from Scikit-learn's `MLPClassifier`, which improves accuracy over the single-layer perceptron by learning **non-linear relationships** in the data.
- **Dataset:**  
  A structured dataset containing various **employability-related attributes**, including:
  - **General Appearance**
  - **Manner of Speaking**
  - **Physical Condition**
  - **Mental Alertness**
  - **Self-Confidence**
  - **Ability to Present Ideas**
  - **Communication Skills**
  - **Student Performance Rating**
- **Feature Engineering:**  
  - **Label encoding** applied to categorical data  
  - **Train-test split** for model evaluation  
- **Evaluation Metrics:**  
  The MLP model is trained and tested, achieving an accuracy score of **XX%** (to be determined after training).  
- **User-Friendly Interface:**  
  - Implemented using **Gradio**, allowing users to input their ratings effortlessly.  
  - Provides real-time employability predictions.  
- **Encouraging Feedback Messages:**  
  - If employable: Motivational message reinforcing their strengths.  
  - If needs improvement: Guidance on how to enhance their skills.  

## **Technology Stack**
- **Programming Language:** Python  
- **Libraries Used:** Pandas, NumPy, Scikit-learn, Gradio  
- **ML Model:** Multi-Layer Perceptron (MLP)  

## **Potential Improvements**
- Hyperparameter tuning (adjusting **hidden layers, activation functions, learning rate**) for better accuracy.  
- Adding **dropout layers** or **batch normalization** to improve training stability.  
- Testing alternative models like **Random Forest, SVM, or Deep Learning with TensorFlow/PyTorch**.  
- Expanding the dataset for better generalization and more accurate predictions.  

## **Conclusion**  
By utilizing an **MLP-based neural network**, this project enhances employability predictions with greater accuracy. The **AI-driven web app** empowers students to assess their readiness for the job market, receive actionable feedback, and take steps toward self-improvement.

---

