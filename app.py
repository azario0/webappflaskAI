
from flask import Flask, request, render_template
import pickle
import pandas as pd
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_values = {key: [float(value)] for key, value in request.form.items()}
    final_features = pd.DataFrame(form_values)
    prediction = model.predict(final_features)

   
    crop_dict = {1: "التوصية هي زراعة الأرز", 2: "التوصية هي زراعة الذرة", 3: "التوصية هي زراعة الجوت", 4: "التوصية هي زراعة القطن", 5: "التوصية هي زراعة جوز الهند", 6: "التوصية هي زراعة البابايا", 7: "التوصية هي زراعة البرتقال",
                    8: "التوصية هي زراعة التفاح", 9: "التوصية هي زراعة الشمام", 10: "التوصية هي زراعة البطيخ", 11: "التوصية هي زراعة العنب", 12: "التوصية هي زراعة المانجو", 13: "التوصية هي زراعة الموز",
                    14: "التوصية هي زراعة الرمان", 15: "التوصية هي زراعة العدس", 16: "التوصية هي زراعة بلاكجرام", 17: "التوصية هي زراعة فول المونج", 18: "التوصية هي زراعة فول العثة",
                    19: "التوصية هي زراعة البازلاء", 20: "التوصية هي زراعة الفاصوليا", 21: "التوصية هي زراعة الحمص", 22: "التوصية هي زراعة القهوة"}
    if prediction[0] in crop_dict:
   
        return render_template('index.html', prediction_text=crop_dict[prediction[0]])
    else:
        return "غير قادر على التوصية بالمحصول المناسب لهذه البيئة"
    
   

if __name__ == "__main__":
    app.run(debug=True)
