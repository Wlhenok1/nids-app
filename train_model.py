# train_model.py — Using KNN Algorithm

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ── STEP 1: Load the dataset ──────────────────────────────────────
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

# ── STEP 2: Fix the labels ────────────────────────────────────────
attack_types = [
    'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'nmap', 'back',
    'teardrop', 'warezclient', 'pod', 'guess_passwd', 'buffer_overflow',
    'warezmaster', 'land', 'imap', 'rootkit', 'loadmodule', 'ftp_write',
    'multihop', 'phf', 'perl', 'spy'
]

df['label'] = df['label'].apply(
    lambda x: 'attack' if str(x).strip().lower() in attack_types else 'normal'
)

print("\nLabel distribution:")
print(df['label'].value_counts())

# ── STEP 3: Convert text columns to numbers ───────────────────────
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'label':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# ── STEP 4: Split into features and labels ────────────────────────
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── STEP 5: Scale the data (VERY important for KNN) ──────────────
# KNN measures distance between points
# Without scaling, large numbers dominate small ones unfairly
# Example: src_bytes (0-99999) would overpower duration (0-10)
print("\nScaling data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── STEP 6: Train the KNN model ───────────────────────────────────
# n_neighbors=5 means it looks at 5 nearest neighbors to vote
# n_jobs=-1 means use all CPU cores to speed it up
print("Training KNN model...")
print("⚠️  KNN is slower than Random Forest — please wait...")
model = KNeighborsClassifier(
    n_neighbors=5,   # look at 5 nearest neighbors
    metric='euclidean',  # measure straight-line distance
    n_jobs=-1        # use all CPU cores
)
model.fit(X_train, y_train)

# ── STEP 7: Test accuracy ─────────────────────────────────────────
print("Testing model...")
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# ── STEP 8: Save everything ───────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(list(df.drop('label', axis=1).columns), f)

# Save scaler too — needed for predictions in app.py
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ KNN Model saved! Now run: python app.py")