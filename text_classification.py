import mojimoji
import neologdn
import MeCab
def normalize_text(text):
    result = mojimoji.zen_to_han(text, kana=False)
    result = neologdn.normalize(result)
    return result


def text_to_words(text):
    m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    m.parse('')
    # 사전정의
    text = normalize_text(text)
    m_text = m.parse(text)
    basic_words = []
    # mecab
    m_text = m_text.split('\n')
    for row in m_text:
        word = row.split("\t")[0]
        if word == 'EOS':
            break
        else:
            pos = row.split('\t')[1]
            slice_ = pos.split(',')
            parts = slice_[0]
            if parts == '기호':
                if word != '。':
                    continue

                basic_words.append(word)
            elif slice_[0] in ('형용사', '동사'):
                    basic_words.append(slice_[-3])

            elif slice_[0] in ('명사', '부사'):
                basic_words.append(word)

    basic_words = ' '.join(basic_words)
    return basic_words
