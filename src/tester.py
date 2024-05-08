from retriever import Vectorizer

def main():
    vectorizer_path = None
    doc_ret = Vectorizer(bigram=False, vectorizer_path=vectorizer_path)
    docs = ["I like dogs",
            "Dogs like dogs",
            "I like cats"]
    
    doc_ret.fit(docs)
    doc_ret.save_vectorizer("vectorizer.pkl")
    # matrix = doc_ret.transform(docs)
    # print(doc_ret.get_feature_names_out())
    # print(matrix.toarray())

if __name__ == '__main__':
    main()
