import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import pearsonr
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ==========================================
# SECTION 1: DATA LOADING & PREPARATION
# ==========================================

# defining the absolute path where i saved the file
file_path = "/Users/kadhir/Downloads/season_shots.json"

# loading the JSON data into a Python list of dictionaries
with open(file_path, "r") as f:
    raw_data = json.load(f)

# converting the list into a structured Pandas DataFrame
df = pd.DataFrame(raw_data)

# inspecting the dimensions, first few rows, and data types
print(f"DataFrame Shape: {df.shape}\n")
print("First 5 rows:")
print(df.head(), "\n")
print("Data Types:")
print(df.dtypes)

# ==========================================
# SECTION 2: DATA TYPE CONVERSIONS
# ==========================================

# converting continuous variables to floats
float_cols = ['X', 'Y', 'xG']
df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')

# converting whole counts to nullable integers
int_cols = ['minute', 'h_goals', 'a_goals']
df[int_cols] = df[int_cols].apply(pd.to_numeric, errors='coerce').astype('Int64')

# verifying the changes
print("\nUpdated Data Types:")
print(df[['X', 'Y', 'xG', 'minute', 'h_goals', 'a_goals']].dtypes)

# ==========================================
# SECTION 3: FEATURE ENGINEERING - DISTANCE
# ==========================================

# calculating distance using pythagorean theorem
df['distance'] = np.sqrt((1 - df['X'])**2 + (0.5 - df['Y'])**2)

# printing the first few rows of our relevant columns to verify
print(df[['X', 'Y', 'distance']].head())

# ==========================================
# SECTION 4: FEATURE ENGINEERING - SHOT ANGLE
# ==========================================

# defining the two goalpost Y-values as constants
y1 = 0.5 - (7.32 / 68) / 2
y2 = 0.5 + (7.32 / 68) / 2

# calculating the horizontal distance (shared for both posts)
horizontal_dist = 1 - df['X']

# calculating two separate angles using np.arctan2
df['angle_to_post1'] = np.arctan2(y1 - df['Y'], horizontal_dist)
df['angle_to_post2'] = np.arctan2(y2 - df['Y'], horizontal_dist)

# calculating the final angle column as the absolute difference
df['angle_rad'] = np.abs(df['angle_to_post1'] - df['angle_to_post2'])

# converting it to degrees so it's easier to read
df['angle_deg'] = np.degrees(df['angle_rad'])

# printing the columns to inspect the intermediate and final steps
print("\nInspecting the angle calculations:")
print(df[['X', 'Y', 'angle_to_post1', 'angle_to_post2', 'angle_rad', 'angle_deg']].head())

# ==========================================
# SECTION 5: TARGET VARIABLE (IS_GOAL)
# ==========================================

print(df['result'].unique())


# creating the binary target variable (goal = 1, everything else = 0)
df['is_goal'] = (df['result'] == 'Goal').astype(int)

# verifying the mapping worked correctly across all unique result types
print("\nChecking the mapping:")
print(df.groupby('result')['is_goal'].value_counts())
print(df['is_goal'].value_counts())
print(df['is_goal'].value_counts(normalize=True))

# ==========================================
# SECTION 6: DATA CLEANING (FILTERING)
# ==========================================

print("\nUnique Shot Types:")
print(df['shotType'].unique())

print("\nUnique Situations:")
print(df['situation'].unique())

# filtering out penalties
df = df[df['situation'] != 'Penalty'].copy()

# checking the new row count
print(f"\nRows left after removing penalties: {len(df)}")

# checking the new class balance (as percentages)
print("\nNew is_goal class balance (%):")
print(df['is_goal'].value_counts(normalize=True) * 100)

# ==========================================
# SECTION 7: CATEGORICAL ENCODING
# ==========================================

# specifying the categorical columns to encode
categorical_cols = ['shotType', 'situation']

# applying one-hot encoding
# dtype=int ensures outputs to be 1s and 0s instead of true/false
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# printing the new list of columns to verify original two are gone + new one-hot encoded columns are present
print("\nColumns after One-Hot Encoding:")
print(df.columns.tolist())

# ==========================================
# SECTION 8: ML DATASET CREATION (X & y)
# ==========================================

# define features (X) and target (y)
feature_cols = [
    'distance', 
    'angle_deg', 
    'shotType_LeftFoot', 
    'shotType_OtherBodyPart', 
    'shotType_RightFoot', 
    'situation_FromCorner', 
    'situation_OpenPlay', 
    'situation_SetPiece'
]

X = df[feature_cols].copy()
y = df['is_goal'].copy()

# final check for missing values
print("\nMissing values in X:")
print(X.isnull().sum())

# printing shapes to verify
print(f"\nShape of X: {X.shape}")
print(f"Shape of y: {y.shape}")



# ==========================================
# SECTION 9: TRAIN / TEST SPLIT
# ==========================================


# splitting the data (80% training, 20% testing)
# stratify=y guarantees the exact same proportion of goals in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# printing the shapes to verify the split was successful
print(f"Training data (X_train): {X_train.shape[0]} shots")
print(f"Testing data (X_test): {X_test.shape[0]} shots")

# checking the class balance in both sets
print("\nClass balance in Training Set (%):")
print(y_train.value_counts(normalize=True) * 100)

print("\nClass balance in Testing Set (%):")
print(y_test.value_counts(normalize=True) * 100)


# ==========================================
# SECTION 10: MODEL TRAINING (LOGISTIC REGRESSION)
# ==========================================

# initializing the model
# max_iter=1000 ensures the underlying math has enough iterations to converge
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# training the model on the hidden training data
lr_model.fit(X_train, y_train)

# predicting the probabilities (xG) on the test data
# predict_proba returns [prob_no_goal, prob_goal] so slicing [:, 1] to keep just prob_goal
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

# printing the first 5 predicted probabilities to verify it worked
print("\nFirst 5 Predicted xG values (Logistic Regression):")
print(y_pred_proba[:5])



# ==========================================
# SECTION 11: MODEL EVALUATION
# ==========================================

# calculating log-loss (lower is better, heavily penalizes confident mistakes)
ll_score = log_loss(y_test, y_pred_proba)

# calculating ROC-AUC (closer to 1.0 is better, measures ability to rank goals higher than misses)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# printing the final metrics
print("\n--- Logistic Regression Performance ---")
print(f"Log-Loss: {ll_score:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")


# ==========================================
# SECTION 12: MODEL TRAINING (TREE-BASED)
# ==========================================

# importing and initializing the model
# random_state=42 ensures reproducibility
tree_model = HistGradientBoostingClassifier(random_state=42)

# fitting the model on the training data
tree_model.fit(X_train, y_train)

# predicting the probabilities on the test data
# using [:, 1] to isolate the probability of a goal
tree_pred_proba = tree_model.predict_proba(X_test)[:, 1]

# evaluating the model
tree_ll_score = log_loss(y_test, tree_pred_proba)
tree_roc_auc = roc_auc_score(y_test, tree_pred_proba)

# printing the final metrics to compare against Logistic Regression
print("\n--- HistGradientBoosting Performance ---")
print(f"Log-Loss: {tree_ll_score:.4f}")
print(f"ROC-AUC:  {tree_roc_auc:.4f}")


# ==========================================
# SECTION 13: COMPARING WITH UNDERSTAT'S xG
# ==========================================

# isolating understat's xG values specifically for the test set rows
# passing X_test.index perfectly aligns the 1,887 rows
understat_xg_test = df.loc[X_test.index, 'xG']

# calculating the pearson correlation
# comparing our logistic regression predictions (y_pred_proba) to understat's
correlation, p_value = pearsonr(y_pred_proba, understat_xg_test)

print("\n--- Model Validation ---")
print(f"Pearson Correlation with Understat xG: {correlation:.4f}")


# ==========================================
# SECTION 14: FULL SEASON PREDICTIONS
# ==========================================

# predicting the probability of a goal for every single shot in the dataset (X)
# using [:, 1] to grab the probability of a goal
full_season_preds = lr_model.predict_proba(X)[:, 1]

# attaching these predictions directly to the main DataFrame as a new column
df['predicted_xG'] = full_season_preds

# printing the first 10 rows of relevant columns to verify it worked perfectly
print("\n--- Full Season Predictions Added ---")
print(df[['player', 'result', 'xG', 'predicted_xG']].head(10))


# ==========================================
# SECTION 15: PLAYER AGGREGATION & OVERPERFORMANCE
# ==========================================

# grouping by player and sum up both their actual goals and predicted xG
player_xg = df.groupby('player')[['is_goal', 'predicted_xG']].sum()

# sorting the values so the players with the highest total xG are at the top
player_xg = player_xg.sort_values(by='predicted_xG', ascending=False).reset_index()

# printing the top 15 players
print("\n--- Top 15 Players by Total Predicted xG ---")
print(player_xg.head(15))



# ==========================================
# SECTION 16: TRANSFER FEES & MERGE
# ==========================================

# creating the transfer fee dictionary (£ millions)
transfer_data = {
    'player': [
        'Alexander Isak', 'Hugo Ekitike', 'Benjamin Sesko', 'Viktor Gyokeres',
        'João Pedro', 'Eberechi Eze', 'Mohammed Kudus', 'Nick Woltemade',
        'Matheus Cunha', 'Bryan Mbeumo', 'Liam Delap', 'Mathis Cherki',
        'Igor Jesus', 'Yoane Wissa'
    ],
    'transfer_fee_m': [
        125.0, 69.0, 66.3, 55.0,
        55.0, 60.0, 55.0, 65.0,
        62.5, 65.0, 30.0, 30.45,
        10.0, 55.0
    ]
}

# converting to a lookup DataFrame
fees_df = pd.DataFrame(transfer_data)

# merging with your player_xg table on the 'player' column
analysis_df = pd.merge(player_xg, fees_df, on='player', how='inner')

# calculating xG over/underperformance (Actual Goals minus Predicted xG)
analysis_df['xG_diff'] = analysis_df['is_goal'] - analysis_df['predicted_xG']

# sorting by transfer fee descending
analysis_df = analysis_df.sort_values(by='transfer_fee_m', ascending=False).reset_index(drop=True)

# displaying the final merged table
print("\n--- Merged Transfer Value vs. Performance Table ---")
print(analysis_df[['player', 'transfer_fee_m', 'is_goal', 'predicted_xG', 'xG_diff']])
# searching for matching names in the original dataset
#print(df[df['player'].str.contains('Cherki', case=False, na=False)]['player'].unique())

# ==========================================
# SECTION 18: SHOT MAP SETUP & PENALTY FILTER
# ==========================================

# identifying implied penalty rows where all dummy columns are 0
is_penalty = (
    (df['situation_FromCorner'] == 0) & 
    (df['situation_OpenPlay'] == 0) & 
    (df['situation_SetPiece'] == 0)
)

# filtering for Liam Delap AND exclude penalties
player_shots = df[(df['player'] == 'Liam Delap') & (~is_penalty)].copy()

# re-scaling coordinates to 0-100 for the Opta pitch
player_shots['X_scaled'] = player_shots['X'] * 100
player_shots['Y_scaled'] = player_shots['Y'] * 100

# verifying shot count
print(f"Total non-penalty shots for Liam Delap: {len(player_shots)}")
print(f"Non-penalty goals: {player_shots['is_goal'].sum()}")

# ==========================================
# SECTION 19: DRAWING THE SHOT MAP
# ==========================================


# initializing the pitch object
pitch = VerticalPitch(
    pitch_type='opta', 
    half=True, 
    pitch_color='#ffffff', 
    line_color='#2b2b2b'
)

# splitting shots into goals and non-goals
goals = player_shots[player_shots['is_goal'] == 1]
non_goals = player_shots[player_shots['is_goal'] == 0]

# drawing the pitch
fig, ax = pitch.draw(figsize=(8, 10))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# plotting non-goals (red circles, sized by xG)
pitch.scatter(
    non_goals['X_scaled'], 
    non_goals['Y_scaled'], 
    ax=ax, 
    color='#e74c3c', 
    edgecolors='black',
    alpha=0.6, 
    s=non_goals['predicted_xG'] * 800 + 40,
    label='No Goal'
)

# plotting goals (green stars)
pitch.scatter(
    goals['X_scaled'], 
    goals['Y_scaled'], 
    ax=ax, 
    color='#2ecc71', 
    edgecolors='black',
    marker='*', 
    s=goals['predicted_xG'] * 800 + 150,
    label='Goal'
)

# title + legend
plt.title("Liam Delap — Non-Penalty Shot Map", fontsize=16, fontweight='bold', pad=10)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=11, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()


# ==========================================
# SECTION 20: JOÃO PEDRO SHOT MAP
# ==========================================


# filtering for João Pedro non-penalty shots
pedro_shots = df[(df['player'] == 'João Pedro') & (~is_penalty)].copy()

# scaling coordinates to 0-100
pedro_shots['X_scaled'] = pedro_shots['X'] * 100
pedro_shots['Y_scaled'] = pedro_shots['Y'] * 100

# splitting into goals and non-goals
goals_pedro = pedro_shots[pedro_shots['is_goal'] == 1]
non_goals_pedro = pedro_shots[pedro_shots['is_goal'] == 0]

# drawing pitch on figure 2 / axis 2
pitch_pedro = VerticalPitch(
    pitch_type='opta', 
    half=True, 
    pitch_color='#ffffff', 
    line_color='#2b2b2b'
)

fig2, ax2 = pitch_pedro.draw(figsize=(8, 10))
fig2.patch.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')

# plotting non-goals (red circles)
pitch_pedro.scatter(
    non_goals_pedro['X_scaled'], 
    non_goals_pedro['Y_scaled'], 
    ax=ax2, 
    color='#e74c3c', 
    edgecolors='black',
    alpha=0.6, 
    s=non_goals_pedro['predicted_xG'] * 800 + 40,
    label='No Goal'
)

# plotting goals (green stars)
pitch_pedro.scatter(
    goals_pedro['X_scaled'], 
    goals_pedro['Y_scaled'], 
    ax=ax2, 
    color='#2ecc71', 
    edgecolors='black',
    marker='*', 
    s=goals_pedro['predicted_xG'] * 800 + 150,
    label='Goal'
)

# title + legend
plt.title("João Pedro — Non-Penalty Shot Map", fontsize=16, fontweight='bold', pad=10)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=11, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()


# ==========================================
# SECTION 21: INTERACTIVE PLOTLY SHOT MAP
# ==========================================


# adding readable outcome label
player_shots = player_shots.copy()
player_shots['outcome_label'] = player_shots['is_goal'].map({1: 'Goal', 0: 'No Goal'})

# splitting shots into goals and non-goals
goals = player_shots[player_shots['is_goal'] == 1]
non_goals = player_shots[player_shots['is_goal'] == 0]

# creating blank Plotly figure
fig = go.Figure()

# adding pitch shapes (boundary, penalty box, six-yard box)
pitch_shapes = [
    # half-pitch boundary
    dict(type="rect", x0=0, y0=50, x1=100, y1=100, line=dict(color="#2b2b2b", width=2)),
    # penalty box
    dict(type="rect", x0=20.4, y0=84.3, x1=79.6, y1=100.0, line=dict(color="#2b2b2b", width=1.5)),
    # six-yard box
    dict(type="rect", x0=36.5, y0=94.8, x1=63.5, y1=100.0, line=dict(color="#2b2b2b", width=1.5)),
]

for shape in pitch_shapes:
    fig.add_shape(shape)

# 5. adding non-goals trace (Red circles sized by xG)
fig.add_trace(
    go.Scatter(
        x=non_goals['Y_scaled'],
        y=non_goals['X_scaled'],
        mode='markers',
        name='No Goal',
        marker=dict(
            size=non_goals['predicted_xG'] * 35 + 8,
            color='rgba(231, 76, 60, 0.7)',
            line=dict(width=1, color='black'),
            symbol='circle'
        ),
        customdata=non_goals[['outcome_label', 'predicted_xG']],
        hovertemplate="<b>Outcome:</b> %{customdata[0]}<br>"
                      "<b>xG:</b> %{customdata[1]:.3f}<br>"
                      "<b>Coordinates:</b> (%{x:.1f}, %{y:.1f})"
                      "<extra></extra>"
    )
)

# adding goals trace (Green star sized by xG)
fig.add_trace(
    go.Scatter(
        x=goals['Y_scaled'],
        y=goals['X_scaled'],
        mode='markers',
        name='Goal',
        marker=dict(
            size=goals['predicted_xG'] * 35 + 16,
            color='rgba(46, 204, 113, 0.95)',
            line=dict(width=1.5, color='black'),
            symbol='star'
        ),
        customdata=goals[['outcome_label', 'predicted_xG']],
        hovertemplate="<b>Outcome:</b> %{customdata[0]} ⚽<br>"
                      "<b>xG:</b> %{customdata[1]:.3f}<br>"
                      "<b>Coordinates:</b> (%{x:.1f}, %{y:.1f})"
                      "<extra></extra>"
    )
)

# configuring layout & styling for full-screen
fig.update_layout(
    autosize=True,  # fills the available browser window
    margin=dict(l=20, r=20, t=60, b=40),  # tightens up the white space around the edges
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(
        range=[45, 105], 
        showgrid=False, 
        zeroline=False, 
        visible=False, 
        scaleanchor="x", 
        scaleratio=1  # ensures the pitch doesn't stretch or warp
    ),
    title=dict(text="<b>Liam Delap</b> — Non-Penalty Shot Map", x=0.5),
    legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5)
)

# rendering figure
fig.show()