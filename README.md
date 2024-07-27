# DrQA Implementation

## Paper: Reading Wikipedia to Answer Open-Domain Questions

> [!SUMMARY]
> This system returns answers to questions using Wikipedia articles. The process is two-fold: first, a Document Retriever identifies a small subset of all articles that contain relevant information with regards to the question. Secondly, a Document Reader looks at all paragraphs in the article, then each word/token in each paragraph, and returns the subset/window of words of text that most likely answers the question.
