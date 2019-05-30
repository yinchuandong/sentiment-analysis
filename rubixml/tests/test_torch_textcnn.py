import numpy as np
from numpy.testing import assert_array_equal
from rubixml.torch_textcnn import TextCNNSentimentClassifier
import dill

RANDOM_SEED = 2333
np.random.seed(RANDOM_SEED)


def test_torch_textcnn_sentiment_model():
    """ test torch-based  textcnn sentiment classifier
    Test Params:
        embed_dim=50
        lr=0.01
        drouput=0.5
        nepoch=1
    """
    model = TextCNNSentimentClassifier(embed_dim=50, lr=0.001, dropout=0.5)
    model.fit('./example_data/train.txt', nepoch=1)

    sentences = ['how',
                 'Wow... Loved this place.',
                 'Crust is not good.',
                 'Not tasty and the texture was just nasty.',
                 'Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.',
                 'There was a warm feeling with the service and I felt like their guest for a special treat.']

    last_result = model.predict_prob(sentences)
    model.use_best_model()
    best_result = model.predict_prob(sentences)

    with open('.textcnn/best_senti_model.pkl', 'wb') as f:
        dill.dump(model, f)
    
    with open('.textcnn/best_senti_model.pkl', 'rb') as f:
        new_model = dill.load(f)
    
    new_result = new_model.predict_prob(sentences)

    print(last_result)
    print(best_result)
    print(new_result)
    # when nepoch is 1, last_result == best_result == new_result
    assert_array_equal(last_result, best_result)
    assert_array_equal(best_result, new_result)
    return


if __name__ == "__main__":
    test_torch_textcnn_sentiment_model()