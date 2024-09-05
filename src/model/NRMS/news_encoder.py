import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.general.attention.multihead_self import MultiHeadSelfAttention
from src.model.general.attention.additive import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ElementEncoder(torch.nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super(ElementEncoder, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        
        # Khởi tạo word embedding
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)

        # Khởi tạo multi-head self-attention cho tiêu đề và abstract
        self.multihead_self_attention_title = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)
        
        self.multihead_self_attention_abstract = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)
        
        # Additive attention để kết hợp các vector từ
        self.additive_attention_title = AdditiveAttention(config.query_vector_dim,
                                                          config.word_embedding_dim)
        
        self.additive_attention_abstract = AdditiveAttention(config.query_vector_dim,
                                                             config.word_embedding_dim)
        
        # Khởi tạo embeddings cho category và subcategory
        self.category_embedding = nn.Embedding(config.num_categories,
                                               config.category_embedding_dim,
                                               padding_idx=0)
        self.subcategory_embedding = nn.Embedding(config.num_categories,
                                                  config.category_embedding_dim,
                                                  padding_idx=0)
        
        # Element encoders cho category và subcategory
        self.category_encoder = ElementEncoder(self.category_embedding,
                                               config.category_embedding_dim,
                                               config.word_embedding_dim)
        self.subcategory_encoder = ElementEncoder(self.subcategory_embedding,
                                                  config.category_embedding_dim,
                                                  config.word_embedding_dim)
        
        # Additive attention cuối cùng để kết hợp tất cả các vector
        self.final_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.word_embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # Xử lý tiêu đề bằng word embedding và multi-head self-attention
        news_title_vector = F.dropout(self.word_embedding(news["title"].to(device)),
                                      p=self.config.dropout_probability,
                                      training=self.training)
        multihead_title_vector = self.multihead_self_attention_title(news_title_vector)
        multihead_title_vector = F.dropout(multihead_title_vector,
                                           p=self.config.dropout_probability,
                                           training=self.training)
        title_vector = self.additive_attention_title(multihead_title_vector)

        # Xử lý abstract bằng word embedding và multi-head self-attention
        news_abstract_vector = F.dropout(self.word_embedding(news["abstract"].to(device)),
                                         p=self.config.dropout_probability,
                                         training=self.training)
        multihead_abstract_vector = self.multihead_self_attention_abstract(news_abstract_vector)
        multihead_abstract_vector = F.dropout(multihead_abstract_vector,
                                              p=self.config.dropout_probability,
                                              training=self.training)
        abstract_vector = self.additive_attention_abstract(multihead_abstract_vector)

        # Xử lý category và subcategory
        category_vector = self.category_encoder(news["category"].to(device))
        subcategory_vector = self.subcategory_encoder(news["subcategory"].to(device))
        
        # Kết hợp các vector tiêu đề, abstract, category, và subcategory
        combined_vectors = torch.stack([title_vector, abstract_vector, category_vector, subcategory_vector], dim=1)
        
        # Vector tin tức cuối cùng sử dụng additive attention
        final_news_vector = self.final_attention(combined_vectors)
        
        return final_news_vector
