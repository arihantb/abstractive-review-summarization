# Abstractive Review Summarization

![Header](src/static/images/image.png)

## Project Overview

This project was the culmination of our final year in machine learning. Our focus was on developing a system for summarizing reviews of products, specifically on Amazon. The goal was not just to extract sentences but to create abstractive summaries that captured the contextual meaning of the reviews. Additionally, we aimed to provide an overall sentiment score, akin to a 5-star rating, and highlight the strengths and weaknesses of the product.

## Project Significance

At the time of this project, product reviews on Amazon were often lengthy and required users to sift through multiple opinions to form an informed decision. Our abstractive summarization approach, along with sentiment scoring and key feature identification, aimed to simplify this process. By providing concise summaries and sentiment scores, users could quickly gauge whether a product was suitable for their needs.

## Technologies Used

- Attentional Sequence-to-Sequence RNN
- Encoder-Decoder
- Pointer Generator Network
- FASTApi.

## Project Workflow

1. **Data Collection:** We scraped product reviews from Amazon.com to create a diverse and comprehensive dataset.
2. **Data Preprocessing:** The collected data underwent extensive cleaning and structuring to ensure its suitability for training.
3. **Model Training:** The heart of the project involved training our Attentional Sequence-to-Sequence RNN model, integrating Encoder-Decoder capabilities and a Pointer Generator Network to enhance abstraction.
4. **Evaluation:** The model's performance was assessed using various metrics, ensuring the generated summaries accurately reflected the sentiments and key aspects of the reviews.
5. **Deployment:** To make the solution accessible to users, we utilized FASTApi for hosting the model online.

## How to Use

- Download the `.h5` model files from [this](https://www.dropbox.com/scl/fi/x0uk3uvmatd3fpcnwwoq4/model_files.rar?rlkey=9wasxapv7vcj70p70qptpxt49&dl=0) Dropbox link.
- Download the `Text-Summarization.ipynb` file.
- Run the above `ipynb` file with the path of the above downloaded models.

## Conclusion

The Abstractive Review Summarization project addresses the challenge of information overload in product reviews. By leveraging advanced machine learning techniques, we aimed to empower users with concise, meaningful summaries, aiding in their decision-making process. The combination of abstractive summarization, sentiment analysis, and feature highlighting sets this project apart in providing a comprehensive overview of product reviews.

## Acknowledgements

- [Text Summarization using LSTM](https://www.kaggle.com/code/singhabhiiitkgp/text-summarization-using-lstm/notebook)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

## Contributors

- [arihantb](https://github.com/arihantb): [arihantbedagkar@gmail.com](mailto:arihantbedagkar@gmail.com)
- [rohan-pednekar](https://github.com/rohan-pednekar): [developer.rohan.pednekar@gmail.com](mailto:developer.rohan.pednekar@gmail.com)

## License

This project is licensed under MIT License. See the [LICENSE](LICENSE) file for details.
