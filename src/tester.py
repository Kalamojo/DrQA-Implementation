from retriever import DocumentRetriever

def main():
    doc_ret = DocumentRetriever(bigram=False)
    docs = ["I like dogs",
            "Dogs like dogs",
            "I like cats"]
    
    doc_ret.fit_documents(documents=docs)
    matrix = doc_ret.get_vectors(documents=docs)
    print(doc_ret.vectorizer.get_feature_names_out())
    print(matrix.toarray())

if __name__ == '__main__':
    main()
