import os
from sentence_transformers import SentenceTransformer


class Embeddings(object):
    def __init__(self, embedding_model_name: str = None):
        if embedding_model_name is None:
            embedding_model_name = os.path.join(self.get_model_path(), "models/bge-large-zh")
        else:
            model_path = os.path.join(self.get_model_path(), "models/{}".format(embedding_model_name))
            if os.path.exists(model_path):
                embedding_model_name = model_path
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def get_model_path(self):
        current_file_path = os.path.abspath(__file__)
        project_dir = current_file_path.split("core")[0]
        return project_dir

    def get_embedding_model(self):
        return self.embedding_model


if __name__ == '__main__':
    conan_embedding = Embeddings("Conan-embedding-v1").get_embedding_model()
    bge_embedding = Embeddings().get_embedding_model()
    a = bge_embedding.encode("今天没雨，客厅窗户也开吧", normalize_embeddings=True, convert_to_tensor=True)
    b = conan_embedding.encode("今天没雨，客厅窗户也开吧", normalize_embeddings=True, convert_to_tensor=True)
    pass
