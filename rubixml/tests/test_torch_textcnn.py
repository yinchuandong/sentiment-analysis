# %%
from rubixml.torch_textcnn
model = TextCNNModel(50, 0.001, 0.5)

model.fit('./raw_data/train.txt')

# %%
print('-----------------------------------------------------------------------')
# model.use_best_model()
sentences = ['how',
             'Wow... Loved this place.',
             'Crust is not good.',
             'Not tasty and the texture was just nasty.',
             'Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.',
             'There was a warm feeling with the service and I felt like their guest for a special treat.']

model.predict_prob(sentences)


# %%
print('-----------------------------------------------------------------------')
model.use_best_model()
sentences = ['how',
             'Wow... Loved this place.',
             'Crust is not good.',
             'Not tasty and the texture was just nasty.',
             'Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.',
             'There was a warm feeling with the service and I felt like their guest for a special treat.']

model.predict_prob(sentences)
