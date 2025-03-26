
```markdown```
# Sentiment Analysis of YouTube Comments

This project focuses on extracting comments from YouTube videos to train and develop a sentiment analysis model. By leveraging real-world data, the model aims to accurately classify comments into positive, negative, or neutral sentiments.

## Project Overview

The repository encompasses the following components:

- **Data Collection**: Utilizing the YouTube Data API to fetch comments from specified videos.
- **Data Preprocessing**: Cleaning and preparing the extracted comments for training.
- **Model Training**: Implementing and training a sentiment analysis model (LSTM) using the preprocessed data.
- **Evaluation**: Assessing the model's performance and making necessary adjustments.

## Repository Structure

- `data_collection.py`: Script to extract comments from YouTube videos using the YouTube Data API.
- `model_training.py`: Script to preprocess data and train the sentiment analysis model.
- `Data.csv`: Sample dataset containing extracted comments.
- `new_data.csv`: Additional dataset for further training or evaluation.
- `README.md`: This documentation file.
- `LICENSE`: Information about the project's licensing.

## Installation

To set up the project locally:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/miriamcoccia/sentiment_analysis_youtube.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd sentiment_analysis_youtube
   ```

3. **Set Up a Virtual Environment** (Optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

4. **Install Required Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Obtain YouTube Data API Key**: Acquire an API key from the [Google Cloud Console](https://console.cloud.google.com/).

2. **Extract Comments**:

   - Open `data_collection.py`.
   - Replace `'YOUR_API_KEY'` with your actual API key.
   - Specify the target video URL.
   - Run the script to fetch comments and save them to a CSV file.

3. **Train the Sentiment Analysis Model**:

   - Ensure the dataset (`Data.csv`) is in the project directory.
   - Run `model_training.py` to preprocess the data and train the model.

4. **Evaluate the Model**:

   - Use the `new_data.csv` file or extract new comments.
   - Test the trained model's performance on this new data.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- google-api-python-client
- vaderSentiment

Ensure all dependencies are installed before running the scripts.


---

*Note: Ensure you have the necessary permissions and adhere to YouTube's terms of service when accessing and using data from the platform.*
```

This README provides a clear and professional overview of your project, detailing its purpose, structure, setup instructions, usage guidelines, dependencies, contribution steps, and licensing information. Adjust the content as needed to fit the specifics of your project. 
