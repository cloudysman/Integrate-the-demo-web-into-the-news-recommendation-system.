import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'NRMS'
# Currently included model
assert model_name in [
    'NRMS', 'NAML', 'LSTUR', 'DKN', 'HiFiArk', 'TANR', 'Exp1'
]
#model_name = 'NAML' #config for NAML
class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_epochs = 2
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 1000
    batch_size = 128
    learning_rate = 0.0001
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 2  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 101221
    num_subcategories=1+ 284
    num_categories = 1 + 295
    num_entities = 1 + 21842
    num_users = 1 + 711222
    word_embedding_dim = 300
    category_embedding_dim = 150
    subcategory_embedding_dim =100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['title','category','subcategory','abstract'], "record": []}
    # For multi-head self-attention
    subcategory_embedding_dim =100
    num_attention_heads = 15
    num_filters = 300
    window_size = 3


class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 300
    window_size = 3
    long_short_term_method = 'ini'
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5


class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['title', 'title_entities'], "record": []}
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False


class HiFiArkConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    num_pooling_heads = 5
    regularizer_loss_weight = 0.1


class TANRConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1


class Exp1Config(BaseConfig):
    dataset_attributes = {
        # TODO ['category', 'subcategory', 'title', 'abstract'],
        "news": ['category', 'subcategory', 'title'],
        "record": []
    }
    # For multi-head self-attention
    num_attention_heads = 15
    ensemble_factor = 1  # Not use ensemble since it's too expensive