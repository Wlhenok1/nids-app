# train_model.py — Using Random Forest (memory efficient)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("Loading data...")

col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

df = pd.read_csv('KDDTrain+.csv', header=0, names=col_names)
df = df.drop('difficulty', axis=1)

attack_types = [
    'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'nmap', 'back',
    'teardrop', 'warezclient', 'pod', 'guess_passwd', 'buffer_overflow',
    'warezmaster', 'land', 'imap', 'rootkit', 'loadmodule', 'ftp_write',
    'multihop', 'phf', 'perl', 'spy'
]

df['label'] = df['label'].apply(
    lambda x: 'attack' if str(x).strip().lower() in attack_types else 'normal'
)

print("Label distribution:")
print(df['label'].value_counts())

# Encode text columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'label':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Random Forest...")
# Use fewer trees to save memory on Render
model = RandomForestClassifier(
    n_estimators=50,   # reduced from 100 to save memory
    max_depth=20,      # limit depth to save memory
    random_state=42,
    n_jobs=1           # use 1 core only on free plan
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

# No scaler needed for Random Forest
import numpy as np
scaler = None
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved! Now run: python app.py")