import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
data_path = 'data.csv'  # Ensure this file is in the same directory as the script
data = pd.read_csv(data_path)

st.title('MS-MARCO Dataset Dashboard')

# Display classification results
st.header('Classification Results')
classification_results = data[['query', 'label', 'pred']]
st.write(classification_results)

# Display clustering results
st.header('Clustering Results')
cluster_counts = data['cluster'].value_counts().reset_index()
cluster_counts.columns = ['cluster', 'count']
fig = px.bar(cluster_counts, x='cluster', y='count', title='Cluster Distribution')
st.plotly_chart(fig)

# Display some example texts for each cluster
st.header('Cluster Examples')
selected_cluster = st.selectbox('Select a cluster', data['cluster'].unique())
cluster_examples = data[data['cluster'] == selected_cluster]['query'].head(10).tolist()
for i, example in enumerate(cluster_examples):
    st.write(f'{i+1}. {example}')

# Display distribution of true labels
st.header('True Label Distribution')
label_counts = data['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']
fig = px.bar(label_counts, x='label', y='count', title='True Label Distribution')
st.plotly_chart(fig)

# Display distribution of predicted labels
st.header('Predicted Label Distribution')
pred_counts = data['pred'].value_counts().reset_index()
pred_counts.columns = ['pred', 'count']
fig = px.bar(pred_counts, x='pred', y='count', title='Predicted Label Distribution')
st.plotly_chart(fig)
